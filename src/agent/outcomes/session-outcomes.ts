/**
 * getSessionOutcomes — one OutcomesV1 computation per root session, shared by
 * every SessionEnd consumer (run receipt + outcome store).
 *
 * Contract: never throws, never runs longer than the total budget (default
 * 10s). Any failure or timeout resolves to `null` and is logged at debug —
 * observability only (SPINE INV-040). The promise is memoized per sessionId
 * so hook registration order is irrelevant: whichever hook asks first starts
 * the computation and both await the same promise. The cache entry is
 * evicted after CONSUMERS reads or after a TTL, whichever comes first.
 *
 * Inputs: session-start git head (session-meta sidecar), end HEAD via a
 * bounded `git rev-parse`, the commit range between them, effect-ledger rows
 * for this sessionId, journal tool events (same source as the facet
 * aggregator), and the last assistant message (same source as the receipt's
 * journalSummary).
 *
 * @module agent/outcomes/session-outcomes
 */

import { debugLog } from '../../utils/debug.js';
import { execFileAsync } from '../../utils/exec-file.js';
import { readSessionMeta, readSessionStart } from '../session/session-meta.js';
import { readEffectsForSession } from '../effect-ledger/store.js';
import { journalRecordsToToolEvents } from '../facets/journal-adapter.js';
import { loadJournalMessages, readJournalRecords } from '../journal/reader.js';
import type { JournalMessage } from '../journal/types.js';
import { computeOutcomes } from './compute-outcomes.js';
import { gitLogRange } from './git-range.js';
import { OutcomesV1Schema, type OutcomesV1 } from './outcomes-v1.schema.js';

export interface SessionOutcomesContext {
  /** Fallback cwd when the session-meta sidecar has none. */
  cwd?: string;
}

/** Number of SessionEnd consumers that read the memo (receipt + store). */
const CONSUMERS = 2;
const TOTAL_TIMEOUT_MS = 10_000;
const HEAD_TIMEOUT_MS = 3_000;
const CACHE_TTL_MS = 60_000;
const SHA40_RE = /^[0-9a-f]{40}$/;

interface Entry {
  promise: Promise<OutcomesV1 | null>;
  reads: number;
  timer: ReturnType<typeof setTimeout>;
}

const cache = new Map<string, Entry>();

/** Test seam: drop every memoized computation. */
export function _resetSessionOutcomesCache(): void {
  for (const e of cache.values()) clearTimeout(e.timer);
  cache.clear();
}

/** Full HEAD sha of `cwd`, or null (not a repo, git missing, timeout). */
export async function readEndHead(cwd: string): Promise<string | null> {
  try {
    const { stdout } = await execFileAsync('git', ['-C', cwd, '--no-optional-locks', 'rev-parse', 'HEAD'], {
      timeout: HEAD_TIMEOUT_MS,
      killSignal: 'SIGTERM',
      maxBuffer: 4096,
    });
    const sha = String(stdout).trim().toLowerCase();
    return SHA40_RE.test(sha) ? sha : null;
  } catch {
    return null;
  }
}

function lastAssistantText(messages: JournalMessage[] | null): string | null {
  if (!messages) return null;
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i]!;
    if (m.role !== 'assistant') continue;
    const text = m.content
      .map((b) => (b.type === 'text' ? b.text : b.type === 'text_ref' ? b.preview : ''))
      .join('');
    if (text.length > 0) return text;
  }
  return null;
}

async function compute(sessionId: string, ctx: SessionOutcomesContext): Promise<OutcomesV1 | null> {
  const cwd = readSessionMeta(sessionId)?.cwd ?? ctx.cwd;
  if (cwd === undefined) return null;
  const start = readSessionStart(sessionId);
  const startHead = start?.headSha ?? null;
  const endHead = await readEndHead(cwd);
  const [gitLog, ledgerEntries] = await Promise.all([
    gitLogRange(cwd, startHead, endHead),
    readEffectsForSession(sessionId).catch((err: unknown) => {
      debugLog(`[session-outcomes] ledger read failed: ${String(err)}`);
      return [];
    }),
  ]);
  const toolEvents = journalRecordsToToolEvents(readJournalRecords(sessionId));
  const outcomes = computeOutcomes({
    cwd,
    startHead,
    endHead,
    gitLog,
    ledgerEntries,
    toolEvents,
    lastAssistantMessage: lastAssistantText(loadJournalMessages(sessionId)),
  });
  const parsed = OutcomesV1Schema.safeParse(outcomes);
  if (!parsed.success) {
    debugLog(`[session-outcomes] schema rejected computed outcomes: ${parsed.error.message}`);
    return null;
  }
  return parsed.data;
}

function bounded(sessionId: string, ctx: SessionOutcomesContext, timeoutMs: number): Promise<OutcomesV1 | null> {
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      debugLog(`[session-outcomes] timed out after ${timeoutMs}ms for ${sessionId}`);
      resolve(null);
    }, timeoutMs);
    timer.unref?.();
    Promise.resolve()
      .then(() => compute(sessionId, ctx))
      .then(
        (v) => resolve(v),
        (err: unknown) => {
          debugLog(`[session-outcomes] computation failed for ${sessionId}: ${String(err)}`);
          resolve(null);
        },
      )
      .finally(() => clearTimeout(timer));
  });
}

/**
 * Memoized OutcomesV1 for a root session. Resolves to null on any failure;
 * never rejects. `ctx.cwd` is the fallback when the session-meta sidecar
 * carries no cwd.
 */
export function getSessionOutcomes(
  sessionId: string,
  ctx: SessionOutcomesContext = {},
  opts: { timeoutMs?: number } = {},
): Promise<OutcomesV1 | null> {
  let entry = cache.get(sessionId);
  if (!entry) {
    const timer = setTimeout(() => cache.delete(sessionId), CACHE_TTL_MS);
    timer.unref?.();
    entry = { promise: bounded(sessionId, ctx, opts.timeoutMs ?? TOTAL_TIMEOUT_MS), reads: 0, timer };
    cache.set(sessionId, entry);
  }
  entry.reads++;
  if (entry.reads >= CONSUMERS) {
    clearTimeout(entry.timer);
    cache.delete(sessionId);
  }
  return entry.promise;
}

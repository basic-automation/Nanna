export default {
  name: "write_file",
  version: "0.1.18",
  output: "memory",
  description: "Write content to a file. BOTH parameters are REQUIRED on every call: file_path AND content (the complete file text). A call without content does nothing and fails. Creates the file if it doesn't exist, overwrites if it does. For files too long to write in one call, use file_buffer (append chunks, then commit) instead. SAFETY: a shrinking rewrite of a file that changed since you last read it returns the file's CURRENT content to merge (not a refusal); a shrinking rewrite that deletes more top-level sections than it keeps is held ONCE with the current content and the removed names — send the same content again to confirm the deletions; blocked if new content is under 30% of the last known-good size (likely truncation), if a .py file would not parse, or if the filename looks like a versioned copy. After each write the cheapest structural check (sh -n / node --check / JSON.parse) runs and its verdict is appended; a full overwrite parks the previous version at <file>.__prev__ for recovery and the richest earlier version at <file>.__best__.",
  parameters: {
    type: "object",
    properties: {
      file_path: { type: "string", description: "REQUIRED. Path to the file to write. Relative paths are resolved against the workspace directory." },
      content: { type: "string", description: "REQUIRED. The complete text to write into the file. Never omit this — a write_file call without content always fails." },
    },
    required: ["file_path", "content"]
  },
  execute: function(input) {
    // Guidance errors are RETURNED, not thrown: a thrown script error reaches
    // the model under five stacked "Execution failed:" prefixes, which small
    // models read as corruption and spiral on.
    function fail(message) {
      return { content: message, success: false };
    }

    // Guard events are logged at INFO so they can be audited from the daemon
    // log after a run. P22 evidence: `grep 'rewrite REMOVED'` over a 4-hour
    // leg returned 0 hits with no way to tell "never fired" from "fired,
    // unlogged" — tool results reach the log only as a ~25-char memory
    // prefix. A safety net you cannot audit is not a safety net. Best-effort.
    function glog(msg) {
      try { Nanna.log("info", msg); } catch (e) { /* logging is optional */ }
    }

    // A path as one shell word for Nanna.exec's `sh -c`: single-quoted with
    // embedded quotes escaped, and a leading `~/` left to the shell as
    // "$HOME". The bridge expands `~` when it writes; quoted literally, the
    // shell looked for a file named `~/x.py` (a valid file then "does NOT
    // parse"), and an apostrophe re-split the words — `Bob's and Ann's
    // notes.md` made a cleanup run `rm -f Bobs and …`.
    function shq(path) {
      var p = String(path);
      var home = "";
      if (p === "~" || p.indexOf("~/") === 0) { home = "\"$HOME\""; p = p.substring(1); }
      if (p === "") return home;
      return home + "'" + p.split("'").join("'\\''") + "'";
    }

    // A refusal the model keeps re-earning has stopped being information.
    //
    // Observed live 2026-07-28 (qwen3.5:9b, 42-feature ladder): the fork guard
    // correctly refused ./minidb.sh because ./minidb already existed, and named
    // the right path and the exact call to make. The model read it and tried
    // ./minidb.sh again — 74 times, 45% of every write attempt in the run. The
    // guard prevented the damage and none of the waste.
    //
    // The explanation is not the problem; its SHAPE is. Each step re-anchors
    // context, so every attempt is effectively the model's first, and a
    // paragraph of reasoning is what a small model skims. So after a couple of
    // identical refusals the message stops explaining and starts commanding:
    // short, imperative, the correct call and nothing else to weigh against it.
    //
    // Counted per path, per workspace, in the same .nanna state dir as the
    // ratchet. Fails OPEN in every direction — a missing or corrupt counter
    // just yields the normal message.
    var REFUSAL_STATE = ".nanna/write_refusals.json";
    var REFUSAL_ESCALATE_AT = 2;
    var REFUSAL_MAX_ENTRIES = 200;
    function refusalBump(key) {
      try {
        var map;
        try { map = JSON.parse(Nanna.readFile(REFUSAL_STATE)); } catch (e2) { map = {}; }
        if (!map || typeof map !== "object") map = {};
        var next = (typeof map[key] === "number" && isFinite(map[key]) ? map[key] : 0) + 1;
        map[key] = next;
        var keys = Object.keys(map);
        if (keys.length > REFUSAL_MAX_ENTRIES) {
          for (var i = 0; i < keys.length - REFUSAL_MAX_ENTRIES; i++) delete map[keys[i]];
        }
        Nanna.writeFile(REFUSAL_STATE, JSON.stringify(map));
        return next;
      } catch (e) {
        return 0;
      }
    }
    // `blunt` replaces `normal` once this path has been refused repeatedly.
    // Keyed "fork:<path>" and SHARED with edit_file on purpose: the model
    // alternated between the two tools on the same doomed path, so a
    // per-tool counter would let it spend the full quota twice over.
    function failEscalating(pathKey, normal, blunt) {
      var n = refusalBump("fork:" + pathKey);
      var msg = n > REFUSAL_ESCALATE_AT ? blunt : normal;
      glog("write_file guard refused (fork, attempt " + n + "): " + pathKey);
      return fail(msg);
    }

    // Anti-erosion ratchet (round-17 lesson): the 30% shrink floor used to be
    // relative to the CURRENT size, so repeated 60-80% rewrites during fault
    // storms compounded (0.7^n) and slowly hollowed files out without ever
    // tripping the guard. The floor base is tracked per workspace in
    // .nanna/write_hiwater.json (.nanna/ is the non-markdown local-state dir —
    // never beside user files, so no sidecar clutter). Each entry stores
    // {hi, last, at, good?, goodAt?, chk?}: `hi` is the high-water mark,
    // `last` is the size write_file ITSELF last left on disk, `good` is the
    // size of the newest version that passed a structural check and `chk` the
    // latest check verdict. While the file EXISTS, `hi` is MONOTONE: an
    // out-of-band change (edit_file, file_buffer, exec, the user) is more
    // evidence of held mass, folded in as max(hi, disk) — never a license to
    // restart the history downward. The 2026-08-08 ornith endurance log
    // showed why the old rule (out-of-band change re-bases hi to disk truth)
    // was a laundering hole: an exec append re-shaped the file, the next
    // write re-based hi 3794→1566, and an 875-byte rewrite that the true
    // history would have refused sailed through — the final artifact kept 8
    // of 37 verified commands.
    //
    // The SHRINK FLOOR, however, anchors on `good` — the LAST EVIDENCED-GOOD
    // size — not on `hi` (P22 Tier 3). Size is not quality: in a long session
    // the largest byte count is usually the most bloated, least-correct
    // draft. Observed 2026-08-10 (ornith leg): a 9768-byte draft that scored
    // 2/42 latched the high-water, and the 30% floor it set (2930 B) refused
    // a legitimate 2830-byte version and left the leg's eventual 16/42 peak
    // (3335 B = 34%) one small edit away from being refused — while the
    // 2952-byte write that actually cost two tests cleared it by 22 bytes.
    // When no structural check has ever passed (no checker applies), the
    // floor falls back to the old hi anchor, so nothing loses protection.
    //
    // Grow-writes are never refused regardless (the refusal requires bytes <
    // current disk size); a deliberate whole-file re-shape has honest doors:
    // delete + recreate (creation re-arms from the new size) or force. Every
    // state operation fails OPEN: a missing or corrupt state file degrades to
    // the old current-size behavior, never blocks a write.
    var HIWATER_STATE = ".nanna/write_hiwater.json";
    // Bound: the state file must stay trivially small over an unbounded
    // daemon lifetime; missions touch tens of files, so 200 entries with
    // least-recently-updated eviction loses nothing real.
    var HIWATER_MAX_ENTRIES = 200;
    // Slash/case normalization plus "./" stripping.
    function hiwaterNormKey(path) {
      var k = path.split("\\").join("/");
      // Case folds only where the filesystem does: on Linux `README.md` and
      // `readme.md` are two files, and one key let a read of either count for both.
      if (Nanna.platform === "win32" || Nanna.platform === "darwin") k = k.toLowerCase();
      while (k.indexOf("./") === 0) k = k.substring(2);
      while (k.indexOf("//") !== -1) k = k.split("//").join("/");
      return k;
    }
    // Canonical ledger key (P22 Tier 3): one file, ONE entry. The old key
    // kept relative and absolute spellings of the same file as separate
    // entries, and the split-brain was observed live (2026-08-10 ornith:
    // 'minidb' hi=9768 next to 'd:/development/.../minidb' hi=3195 — one
    // absolute-path edit silently cut the effective floor from 2930 to 958
    // bytes). Absolute spellings under the workspace root now collapse to
    // the relative form; hiwaterEntryFor still consults the old spelling so
    // pre-existing entries are found and folded forward (healed on the next
    // save). Fails open: no workdir → the old per-spelling behavior.
    function hiwaterKey(path) {
      var k = hiwaterNormKey(path);
      try {
        var wd = Nanna.workdir();
        if (wd) {
          var w = hiwaterNormKey(String(wd));
          if (w.charAt(w.length - 1) !== "/") w += "/";
          if (k.indexOf(w) === 0 && k.length > w.length) k = k.substring(w.length);
        }
      } catch (e) {
        // No workdir — spellings keep their own entries, as before.
      }
      return k;
    }
    // Transient park buffers and recovery copies must never be judged by (or
    // recorded in) cross-call history — they are rewritten wholesale — and
    // the ratchet's own state file guards itself specially (below).
    function hiwaterIsBuffer(key) {
      var buf = ".__buffer__";
      return key.length >= buf.length && key.lastIndexOf(buf) === key.length - buf.length;
    }
    function hiwaterIsPrev(key) {
      var p = ".__prev__";
      return key.length >= p.length && key.lastIndexOf(p) === key.length - p.length;
    }
    // The coverage high-water park (below) is a recovery copy like .__prev__:
    // rewritten wholesale, never judged by cross-call history.
    function hiwaterIsBest(key) {
      var b = ".__best__";
      return key.length >= b.length && key.lastIndexOf(b) === key.length - b.length;
    }
    // Exact path only (root or any /.nanna/ dir), so a real work file with a
    // similar name keeps full ratchet protection.
    function hiwaterIsState(key) {
      if (key === ".nanna/write_hiwater.json") return true;
      var tail = "/.nanna/write_hiwater.json";
      return key.length > tail.length && key.lastIndexOf(tail) === key.length - tail.length;
    }
    function hiwaterExempt(key) {
      return hiwaterIsBuffer(key) || hiwaterIsPrev(key) || hiwaterIsBest(key) || hiwaterIsState(key);
    }
    function hiwaterLoad() {
      try {
        var raw = Nanna.readFile(HIWATER_STATE);
        if (raw) {
          var parsed = JSON.parse(raw);
          if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed;
        }
      } catch (e) {
        // Missing or corrupt state: start fresh.
      }
      return {};
    }
    function hiwaterHi(entry) {
      if (entry && typeof entry.hi === "number" && isFinite(entry.hi) && entry.hi > 0) return entry.hi;
      return 0;
    }
    function hiwaterGood(entry) {
      if (entry && typeof entry.good === "number" && isFinite(entry.good) && entry.good > 0) return entry.good;
      return 0;
    }
    function hiwaterSave(map) {
      try {
        var keys = Object.keys(map);
        if (keys.length > HIWATER_MAX_ENTRIES) {
          keys.sort(function(a, b) {
            return ((map[a] && map[a].at) || 0) - ((map[b] && map[b].at) || 0);
          });
          var evict = keys.length - HIWATER_MAX_ENTRIES;
          for (var i = 0; i < evict; i++) delete map[keys[i]];
        }
        Nanna.writeFile(HIWATER_STATE, JSON.stringify(map));
      } catch (e) {
        // State persistence is best-effort.
      }
    }
    // The entry for a path under BOTH its canonical and its legacy spelling,
    // folded into one: hi takes the max (monotone evidence), everything else
    // follows the fresher entry. When the map is later saved through, the
    // alias entry is dropped — the split-brain heals on first touch.
    function hiwaterEntryFor(map, path) {
      var canon = hiwaterKey(path);
      var alias = hiwaterNormKey(path);
      var a = map[canon] || null;
      var b = alias !== canon ? (map[alias] || null) : null;
      if (b) {
        var merged;
        if (!a) {
          merged = b;
        } else {
          merged = ((b.at || 0) > (a.at || 0)) ? b : a;
          var other = merged === a ? b : a;
          var hiM = hiwaterHi(merged);
          var hiO = hiwaterHi(other);
          if (hiO > hiM) merged.hi = hiO;
          if (hiwaterGood(merged) === 0 && hiwaterGood(other) > 0) {
            merged.good = other.good;
            merged.goodAt = other.goodAt || 0;
            if (merged.chk !== "ok" && merged.chk !== "bad") merged.chk = other.chk;
          }
        }
        delete map[alias];
        map[canon] = merged;
        return merged;
      }
      return a;
    }
    // Sign an acknowledged removal set onto the ratchet entry (P23 structural
    // shrink hold, below). A follow-up write whose removal set signs the same
    // proceeds — the deletions were named and in view when it was re-sent —
    // and every successful write rebuilds the entry from scratch, so the
    // signature clears itself and the hold re-arms for the next removal
    // event. Best-effort: an unrecorded signature costs one more hold, never
    // a write.
    function heldGutPut(path, sig) {
      try {
        var map = hiwaterLoad();
        var key = hiwaterKey(path);
        var entry = hiwaterEntryFor(map, path);
        if (!entry) entry = { hi: 0, last: 0, at: Date.now() };
        entry.heldGut = sig;
        entry.at = Date.now();
        map[key] = entry;
        hiwaterSave(map);
      } catch (e) {
        // Best-effort.
      }
    }

    // USER-DECLARED FILE INVARIANTS (P23). Durable prohibitions the USER
    // stated in chat — "never create, edit or delete anything under tests/" —
    // are registered at plan time as (kind, glob, the verbatim sentence,
    // declaring scope) and consulted here, at the last point a mutation can
    // still be stopped honestly. The refusal quotes the user's own sentence
    // instead of paraphrasing it: a constraint the model cannot recognize as
    // the user's reads as a tool malfunction and gets routed around.
    //
    // Missing, unreadable or malformed registry => NO invariants (fail open,
    // silently) — this can never be the reason a write fails. force does NOT
    // bypass: only the user lifts a constraint, and a model that believes the
    // constraint blocks the mission has ask_user, the wanted escape hatch.
    // `read_only` forbids any mutation of a matching path, `no_create_under`
    // only the creation of a path that does not exist yet; `no_delete` is
    // about deletion and is enforced where deletions happen (exec), not here.
    var INVARIANT_STATE = ".nanna/declared_invariants.json";
    function invariantsLoad() {
      try {
        var raw = Nanna.readFile(INVARIANT_STATE);
        if (!raw) return [];
        var parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== "object") return [];
        var list = parsed.invariants;
        if (!Array.isArray(list)) return [];
        return list;
      } catch (e) {
        return [];
      }
    }
    // Glob match over the SAME canonical spelling the ratchet ledger uses, so
    // one file has one identity here too. `**` crosses directory separators,
    // `*`/`?` stop at one; a wildcard-free glob matches the path itself or
    // anything under it (a user who says "tests/" means the tree). Any regex
    // failure yields NO match — fail open.
    function invariantMatches(glob, canonPath, normPath) {
      try {
        // Canonical on BOTH sides: an absolute glob under the workspace
        // root collapses to the relative form exactly as the path does.
        var g = hiwaterKey(String(glob));
        while (g.length > 1 && g.charAt(g.length - 1) === "/") g = g.substring(0, g.length - 1);
        if (g === "") return false;
        if (g.indexOf("*") === -1 && g.indexOf("?") === -1) {
          return canonPath === g || normPath === g ||
            canonPath.indexOf(g + "/") === 0 || normPath.indexOf(g + "/") === 0;
        }
        var re = "";
        for (var i = 0; i < g.length; i++) {
          var c = g.charAt(i);
          if (c === "*") {
            if (g.charAt(i + 1) === "*") {
              re += "[\\s\\S]*";
              i++;
              if (g.charAt(i + 1) === "/") i++;
            } else {
              re += "[^/]*";
            }
          } else if (c === "?") {
            re += "[^/]";
          } else if ("\\^$.|+()[]{}".indexOf(c) !== -1) {
            re += "\\" + c;
          } else {
            re += c;
          }
        }
        var rx = new RegExp("^" + re + "$");
        return rx.test(canonPath) || rx.test(normPath);
      } catch (e) {
        return false;
      }
    }
    // The refusal text for a path a declared invariant forbids mutating, or
    // "" when nothing applies. `verb` is the tool's own word for the action so
    // the sentence reads as this tool's refusal, not a generic one.
    function invariantRefusal(path, verb) {
      try {
        // The registry itself is one of the rules' paths: rewriting it to
        // "{}" erased every rule the user had declared, with no undo snapshot
        // (.nanna/ is housekeeping). Only lift_invariant, which asks the
        // user, changes it.
        var regKey = hiwaterKey(path);
        var regTail = "/.nanna/declared_invariants.json";
        if (regKey === ".nanna/declared_invariants.json" ||
            (regKey.length > regTail.length && regKey.lastIndexOf(regTail) === regKey.length - regTail.length)) {
          return verb + " REFUSED — " + path + " holds the rules the user declared; nothing was " +
            "written. Only lift_invariant (which asks the user) changes it. Continue with your task.";
        }
        var list = invariantsLoad();
        if (list.length === 0) return "";
        var canon = hiwaterKey(path);
        var norm = hiwaterNormKey(path);
        for (var i = 0; i < list.length; i++) {
          var inv = list[i];
          if (!inv || typeof inv !== "object") continue;
          var kind = typeof inv.kind === "string" ? inv.kind : "";
          if (kind !== "read_only" && kind !== "no_create_under") continue;
          if (typeof inv.glob !== "string") continue;
          if (!invariantMatches(inv.glob, canon, norm)) continue;
          if (kind === "no_create_under") {
            // Only CREATION is forbidden: an existing file under the glob is
            // not this constraint's business. One stat, only on a match.
            var exists = false;
            try { exists = !!Nanna.stat(path); } catch (eS) { exists = false; }
            if (exists) continue;
          }
          var quoted = typeof inv.source === "string" && inv.source !== "" ? inv.source : "";
          var scope = typeof inv.scope === "string" && inv.scope !== "" ? inv.scope : "session";
          glog("write_file guard: " + verb + " REFUSED (declared invariant " + kind + " on '" + inv.glob + "') " + path);
          return verb + " REFUSED — " + path + " is under a path you declared off-limits" +
            (kind === "no_create_under" ? " for NEW files" : "") +
            (quoted === "" ? " (" + kind + " on `" + inv.glob + "`)" : ": \"" + quoted + "\"") +
            ". Nothing was written and the file on disk is unchanged. That is YOUR instruction (declared for " +
            scope + "), not a tool limitation, and it stays in force until you lift it in chat. " +
            "The fix belongs in the artifact you are producing — if something that READS " + path +
            " is failing, change the code it exercises, not " + path + ". If you believe this constraint " +
            "genuinely blocks the goal, call lift_invariant with glob `" + inv.glob + "` — it asks the user and lifts " +
            "the rule only on their yes — instead of working around it.";
        }
        return "";
      } catch (e) {
        return "";
      }
    }

    // WRITE-FAILURE HONESTY (P23). A failed write has a CAUSE and the cause
    // lives at the TAIL of the OS message — exactly what a head truncation
    // eats. Classify BEFORE truncating and keep the tail. "Retry the same
    // call" is advice that only makes sense for a plausibly transient fault:
    // on a permission denial it is a lie the model pays for in identical
    // retries. Parses BOTH the current bridge format ("Failed to write
    // '<path>': <os message>") and one carrying the stable ErrorKind name
    // ("... (kind=PermissionDenied): <os message>"); with neither, it falls
    // back to the OS error NUMBER, which is locale-independent (5 = Windows
    // access denied, 13 = EACCES, 32/33 = sharing/lock violation,
    // 4 = interrupted).
    function writeErrorKind(msg) {
      var m = /\(kind=([A-Za-z]+)\)/.exec(msg);
      if (m) return m[1];
      if (msg.indexOf("(os error 5)") !== -1 || msg.indexOf("(os error 13)") !== -1) return "PermissionDenied";
      if (msg.indexOf("(os error 32)") !== -1 || msg.indexOf("(os error 33)") !== -1) return "SharingViolation";
      if (msg.indexOf("(os error 4)") !== -1) return "Interrupted";
      return "";
    }
    function writeErrorTransient(kind) {
      return kind === "Interrupted" || kind === "TimedOut" || kind === "WouldBlock" ||
        kind === "SharingViolation" || kind === "ResourceBusy";
    }
    // Same 120-char identification width as before, split so the trailing
    // cause survives the cut instead of the head eating the whole budget.
    function preserveCause(msg) {
      if (msg.length <= 120) return msg;
      return msg.substring(0, 80) + " … " + msg.substring(msg.length - 40);
    }
    // "(cause). <advice>" — cause first, then whether a retry can possibly
    // help, then what the failure MEANS. Write protection is a deliberate
    // signal, not an obstacle to route around.
    function writeFailureNote(path, rawErr, retryAdvice) {
      var kind = writeErrorKind(rawErr);
      var shown = preserveCause(rawErr);
      if (kind === "PermissionDenied") {
        // FileStat carries no read-only bit, so the stat distinguishes what it
        // CAN — a protected existing file from a directory from a path whose
        // parent is protected — and the sentence claims only the denial the
        // OS actually reported.
        var what = " The filesystem refused the write, not this tool.";
        try {
          var st = Nanna.stat(path);
          if (st && st.is_dir) {
            what = " " + path + " is a DIRECTORY, not a file — writing to it can never work.";
          } else if (st && st.is_file) {
            what = " " + path + " exists and is write-protected on disk.";
          }
        } catch (eStat) {
          what = " The path cannot even be stat'ed, so the protection is on the file or on the directory holding it.";
        }
        return "(" + shown + ")." + what +
          " Retrying the identical call cannot succeed — nothing transient failed. " +
          "Write protection is usually deliberate: it marks the file as INPUT. Unless the request is " +
          "specifically to change THIS file, leave the protection in place and change the file you are producing instead.";
      }
      if (kind === "" || writeErrorTransient(kind)) {
        return "(" + shown + "). " + retryAdvice;
      }
      return "(" + shown + "). That failure is not transient — retrying the identical call will fail the same way. " +
        "Fix what the cause names (the path, the directory, the disk) and then write again.";
    }

    // Read-recency marks (P22 Tier 3): when did this session last SEE this
    // file's content? Recorded by read_file, by a successful whole-file
    // write here (at that instant the file holds exactly what the model
    // sent), and by the stale-shrink echo below (the echo IS a read). The
    // mark is compared against the file's mtime: any mutation — edit_file,
    // exec redirect, the user — moves mtime past the mark and invalidates
    // it. All I/O fails OPEN.
    var READMARK_STATE = ".nanna/read_marks.json";
    var READMARK_MAX_ENTRIES = 200;
    function readmarkLoad() {
      try {
        var raw = Nanna.readFile(READMARK_STATE);
        if (raw) {
          var parsed = JSON.parse(raw);
          if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return parsed;
        }
      } catch (e) {
        // Missing or corrupt state: start fresh.
      }
      return {};
    }
    function readmarkPut(path) {
      try {
        var map = readmarkLoad();
        map[hiwaterKey(path)] = { at: Date.now() };
        var keys = Object.keys(map);
        if (keys.length > READMARK_MAX_ENTRIES) {
          keys.sort(function(a, b) {
            return ((map[a] && map[a].at) || 0) - ((map[b] && map[b].at) || 0);
          });
          var evict = keys.length - READMARK_MAX_ENTRIES;
          for (var i = 0; i < evict; i++) delete map[keys[i]];
        }
        Nanna.writeFile(READMARK_STATE, JSON.stringify(map));
      } catch (e) {
        // Best-effort.
      }
    }
    // Has the session seen this file's CURRENT content? Three verdicts,
    // because the shrink hold below fires for two DIFFERENT reasons and a
    // reply that states the wrong one is a lie the model then reasons from:
    //   "seen"  — a mark exists and is at least as new as the file's mtime.
    //             Unknown → "seen" (fail open).
    //   "stale" — a mark exists but predates the mtime: the file really did
    //             change since the session last read it.
    //   "never" — no mark at all: the session has never read this file.
    // The hold is identical for both non-"seen" verdicts (you cannot shrink
    // what you have not seen) and so is the way forward, but telling a model
    // its copy went stale when it never had one sends it hunting for a change
    // that never happened — and these replies are read literally by 9B-class
    // local models. stat.modified is whole seconds and truncates, which
    // biases sub-second races toward "seen" — the open direction.
    function readSeenVerdict(path) {
      try {
        var st = Nanna.stat(path);
        if (!st || typeof st.modified !== "number" || !isFinite(st.modified)) return "seen";
        var entry = readmarkLoad()[hiwaterKey(path)];
        var at = entry && typeof entry.at === "number" && isFinite(entry.at) ? entry.at : 0;
        if (at === 0) return "never";
        return at >= st.modified * 1000 ? "seen" : "stale";
      } catch (e) {
        return "seen";
      }
    }

    // Refuse ANY .py content that does not parse — new file or overwrite.
    // Round-6 lesson: gating only valid->invalid transitions let the model
    // create a file BORN broken and then "repair" it with equally broken
    // content forever. Invalid Python on disk is never useful; the error
    // names the line so the write call becomes a fast syntax feedback
    // loop. force=true overrides; ANY checker failure fails OPEN (a
    // missing python interpreter must never block writes). Returns
    // {ran, ok, detail} — `ran` distinguishes a real verdict from a
    // fail-open non-answer, so only genuine passes feed the evidenced-good
    // anchor below.
    function pythonSyntaxCheck(path, nextText) {
      var lower = path.toLowerCase();
      if (lower.length < 3 || lower.lastIndexOf(".py") !== lower.length - 3) return { ran: false, ok: false, detail: "" };
      try {
        var chk = path + ".__chk.py";
        var newTmp = path + ".__chk_new.py";
        Nanna.writeFile(newTmp, nextText);
        Nanna.writeFile(chk,
          "import ast, sys\n" +
          "try:\n" +
          "    ast.parse(open(sys.argv[1], encoding='utf-8').read())\n" +
          "    print('NEW_OK')\n" +
          "except SyntaxError as e:\n" +
          "    print('NEW_BAD line ' + str(e.lineno) + ': ' + str(e.msg))\n");
        var cmd = "python " + shq(chk) + " " + shq(newTmp) + "; rc=$?; rm -f " + shq(chk) + " " + shq(newTmp) + "; exit $rc";
        var result = Nanna.exec(cmd, null, 30);
        var out = result && result.stdout ? result.stdout : "";
        var bad = out.indexOf("NEW_BAD");
        if (bad !== -1) {
          var detail = out.substring(bad + 8);
          var nl = detail.indexOf("\n");
          if (nl !== -1) detail = detail.substring(0, nl);
          if (detail.length > 160) detail = detail.substring(0, 160);
          return { ran: true, ok: false, detail: detail };
        }
        if (out.indexOf("NEW_OK") !== -1) return { ran: true, ok: true, detail: "" };
        return { ran: false, ok: false, detail: "" };
      } catch (e) {
        return { ran: false, ok: false, detail: "" };
      }
    }

    // Post-mutation structural check (P22 Tier 3): after any write, the
    // cheapest applicable check runs on what actually landed and its verdict
    // is appended to the result AS A SENTENCE — never a gate. Evidence
    // (gemma, 2026-08-10): a partial-overlap edit left a dangling `else`;
    // the broken file exits 2, which is exactly what the first test asserts,
    // so the surviving failure said "should print usage" and the model spent
    // 31 minutes editing help text on a file that had not parsed for any of
    // them. The truth was one `sh -n` away. A pass also records the size as
    // the ratchet's evidenced-good anchor. Every failure path (no checker,
    // checker missing, timeout) yields NO verdict at all — fail OPEN.
    function structuralCheckKind(path, contentText) {
      var p = path.split("\\").join("/").toLowerCase();
      var base = p.split("/").pop();
      function hasExt(e) { return base.length > e.length && base.lastIndexOf(e) === base.length - e.length; }
      if (hasExt(".json") || hasExt(".geojson")) return "json";
      // .bash (and bash shebangs below) route to bash -n: on hosts where
      // /bin/sh is dash, valid bash ([[ ]], arrays, process substitution)
      // fails sh -n and the verdict would cry wolf on a correct file
      // (ultrareview on PR #224).
      if (hasExt(".bash")) return "bash";
      if (hasExt(".sh")) return "sh";
      if (hasExt(".js") || hasExt(".mjs") || hasExt(".cjs")) return "node";
      if (hasExt(".py")) return "py";
      // Extensionless artifacts (./minidb) are the common mission shape:
      // classify by shebang instead of the name. "sh" must not match fish/
      // pwsh/zsh/csh — those are not POSIX sh and sh -n would cry wolf.
      if (base.indexOf(".") === -1 && typeof contentText === "string" && contentText.indexOf("#!") === 0) {
        var nl = contentText.indexOf("\n");
        var line1 = nl === -1 ? contentText : contentText.substring(0, nl);
        if (line1.indexOf("python") !== -1) return "py";
        if (line1.indexOf("node") !== -1) return "node";
        if (line1.indexOf("bash") !== -1) return "bash";
        if (line1.indexOf("fish") === -1 && line1.indexOf("pwsh") === -1 &&
            line1.indexOf("zsh") === -1 && line1.indexOf("csh") === -1 &&
            line1.indexOf("sh") !== -1) return "sh";
      }
      return null;
    }
    // Runs the check. Returns {ok, tool, detail} or null for no verdict.
    // 15s cap: these checks parse without executing, so they are near-
    // instant; the cap only bounds interpreter cold start and must stay far
    // under exec's 180s engine deadline so a check can never orphan a child.
    function runStructuralCheck(kind, path, contentText) {
      if (kind === "json") {
        if (typeof contentText !== "string") return null;
        try {
          JSON.parse(contentText);
          return { ok: true, tool: "JSON.parse", detail: "" };
        } catch (eJ) {
          var jd = String(eJ && eJ.message ? eJ.message : eJ);
          if (jd.length > 160) jd = jd.substring(0, 160);
          return { ok: false, tool: "JSON.parse", detail: jd };
        }
      }
      var cmd = null;
      var toolName = null;
      if (kind === "sh") { cmd = "sh -n " + shq(path); toolName = "sh -n"; }
      else if (kind === "bash") { cmd = "bash -n " + shq(path); toolName = "bash -n"; }
      else if (kind === "node") { cmd = "node --check " + shq(path); toolName = "node --check"; }
      else if (kind === "py") { cmd = "python -c 'import ast,sys; ast.parse(open(sys.argv[1], encoding=\"utf-8\").read())' " + shq(path); toolName = "python ast"; }
      if (!cmd) return null;
      try {
        var r = Nanna.exec(cmd, null, 15);
        if (!r) return null;
        if (r.success) return { ok: true, tool: toolName, detail: "" };
        var err = (r.stderr || r.stdout || "");
        if (r.code === 127 || err.indexOf("command not found") !== -1) return null; // checker absent
        err = err.split("\r").join("").split("\n").join(" ");
        while (err.indexOf("  ") !== -1) err = err.split("  ").join(" ");
        if (err.length > 200) err = err.substring(0, 200);
        if (err === "" || err === " ") err = "exit code " + r.code;
        return { ok: false, tool: toolName, detail: err };
      } catch (eX) {
        return null;
      }
    }
    function structSentence(path, verdict, prevChk) {
      if (!verdict) return "";
      if (verdict.ok) {
        if (prevChk === "bad") return " STRUCTURE: " + path + " parses again (" + verdict.tool + ") — the earlier syntax break is fixed.";
        return " STRUCTURE: the file parses (" + verdict.tool + ").";
      }
      var history = "";
      if (prevChk === "ok") history = " It parsed BEFORE this write — this write introduced the break.";
      else if (prevChk === "bad") history = " It did not parse before this write either.";
      return " STRUCTURE: " + path + " does NOT parse (" + verdict.tool + "): " + verdict.detail + "." + history +
        " This is information, not a block — the file holds exactly what you sent. Fix that line with edit_file.";
    }
    // LITERAL-ESCAPE VERDICT (P23), shell-checked files only. The shipped
    // repair above catches the whole-file flattening (>=2 literal \n and NO
    // real newlines). The PARTIAL case survives it: the file has real line
    // breaks, but one physical line carries a flattened block behind a '#'.
    // sh gives comments no escape semantics, so the file parses cleanly and
    // the hidden code silently never runs — the parse verdict says "ok" and
    // means nothing. A SENTENCE, never a gate and never a repair: converting
    // on a heuristic would ACTIVATE code the author may not have meant to
    // run. Comment-scoped, so printf/awk lines with legitimate literal \n
    // never trip it, and the >=2 bound is inherited from the repair (two
    // literal \n describe three or more lines). Fails open.
    function escapedCommentNote(kind, text) {
      if (kind !== "sh" && kind !== "bash") return "";
      try {
        if (typeof text !== "string" || text.indexOf("\\n") === -1) return "";
        var re = /^[ \t]*#[^\n]*$/gm;
        var m;
        var scanned = 0;
        var lineNo = 1;
        while ((m = re.exec(text)) !== null) {
          while (scanned < m.index) {
            if (text.charAt(scanned) === "\n") lineNo++;
            scanned++;
          }
          var hits = m[0].split("\\n").length - 1;
          if (hits >= 2) {
            return " NOTE: line " + lineNo + " is a comment carrying " + hits +
              " literal \\n sequences — flattened code may be hiding behind it;" +
              " if you meant newlines, rewrite the file.";
          }
          if (m.index === re.lastIndex) re.lastIndex++;
        }
        return "";
      } catch (e) {
        return "";
      }
    }

    // Accept multiple parameter name variants from different models
    var filePath = input.file_path || input.filePath || input.path || input.file || input.filename;
    // Undefined-chain, NOT || — an explicit content:"" is a legitimate
    // empty-file write (e.g. a package __init__.py) and must not be
    // misreported as "missing content" (verify-round finding).
    var fileContent = input.content;
    if (fileContent === undefined || fileContent === null) fileContent = input.text;
    if (fileContent === undefined || fileContent === null) fileContent = input.data;
    if (fileContent === undefined || fileContent === null) fileContent = input.new_content;
    if (fileContent === undefined || fileContent === null) fileContent = input.file_content;
    if (!filePath && (fileContent === undefined || fileContent === null)) {
      return fail("write_file failed: you must pass BOTH file_path AND content. Call it again like: write_file(file_path=\"D:/path/to/file.py\", content=\"<the complete file text>\")");
    }
    if (!filePath) {
      return fail("write_file failed: missing file_path. Nothing was written. Call it again with BOTH file_path (the destination path) AND content.");
    }
    if (fileContent === undefined || fileContent === null) {
      return fail("write_file failed: missing content. Nothing was written. Call write_file again with file_path=\"" + filePath + "\" AND content set to the COMPLETE file text.");
    }
    if (typeof filePath !== "string") filePath = String(filePath);
    if (typeof fileContent !== "string") fileContent = String(fileContent);

    // Writing the ratchet's own bookkeeping is always confusion — and wiping
    // it would silently disarm the erosion guard. Calm, self-describing
    // refusal so the model moves on instead of "repairing" internals.
    if (!input.force && hiwaterIsState(hiwaterKey(filePath))) {
      return fail("write_file skipped: " + filePath + " is write_file's internal bookkeeping. It maintains itself, is healthy, and never needs manual repair. Your own files are unaffected. Continue with your actual task.");
    }

    // The user's own declared prohibitions come FIRST — before the syntax
    // gate, which writes .__chk temp files beside the target, and before any
    // parking or repair. Nothing has been touched at this point.
    var invBlock = invariantRefusal(filePath, "WRITE");
    if (invBlock !== "") return fail(invBlock);

    // Double-escaped content — REPAIRED, not refused. Observed live
    // (lfm2.5): the model emitted "#!/bin/sh\ncase $1 in" with the
    // backslash-n as TWO LITERAL CHARACTERS, so the "script" landed as one
    // physical line and could never execute; every downstream acceptance
    // check then failed on behaviour, hiding the real cause.
    //
    // Refusing was tried first and is WORSE: the model cannot see its own
    // escaping (the defect is in how its output is serialized, not in its
    // intent), so it resent byte-identical content three times, then began
    // SHRINKING the file to appease the guard — heading straight for the
    // truncation ratchet. A guard the model cannot satisfy is a wedge.
    //
    // The signature is unambiguous and needs no size threshold: content
    // carrying two or more literal \n sequences describes three or more
    // lines, so a complete absence of real newlines means the escaping was
    // lost in transit, not that the author wanted one line. JSON-family
    // files are exempt — \n inside a string is exactly how JSON encodes a
    // newline, by spec. force=true writes bytes through untouched.
    function escapedNewlineCount(path, text) {
      if (text.indexOf("\n") !== -1) return 0; // real newlines present — fine
      var lower = path.toLowerCase();
      if (/\.(json|jsonl|ndjson|geojson)$/.test(lower)) return 0;
      return text.split("\\n").length - 1;
    }
    var unescapedNewlines = 0;
    if (!input.force) {
      unescapedNewlines = escapedNewlineCount(filePath, fileContent);
      if (unescapedNewlines >= 2) {
        fileContent = fileContent.split("\\n").join("\n");
      } else {
        unescapedNewlines = 0;
      }
    }

    var bytes = fileContent.length;

    // The pre-write definition pass. Computed once (when the structural hold
    // below needs it) and reused by the rewrite-note after the write, so each
    // side is parsed exactly once per call.
    var preOldBodies = null;
    var preNewBodies = null;

    // Safety checks BEFORE writing. Three invariants from the adversarial
    // verify round still hold: a write that does not shrink the CURRENT
    // disk file can never erode it, so it is never refused; a file that no
    // longer exists has nothing left to protect — creation is always allowed
    // and re-arms the ratchet from the new size; and every state failure
    // degrades to the old behavior, never to a block.
    var existingSize = 0;
    var fileExists = false;
    var existsUnknown = false;
    var existing;
    if (!input.force) {
      try {
        existing = Nanna.readFile(filePath);
        if (existing !== undefined && existing !== null) {
          fileExists = true;
          existingSize = existing.length;
        }
      } catch (e) {
        // "os error 2"/"os error 3" = the file genuinely doesn't exist. Any
        // OTHER read failure (sharing violation, non-UTF-8) means the file is
        // probably THERE but unreadable: fail open on the guard, but flag it
        // so the ratchet state is left untouched (verify finding: a transient
        // lock must not reset the mark to the new small size).
        // Parenthesized form ONLY: the bridge embeds io::Error display as
        // "...(os error N)", and a bare "os error 3" substring also matches
        // "(os error 32)" — the sharing-violation case this flag exists for.
        var readErr = String(e);
        if (readErr.indexOf("(os error 2)") === -1 && readErr.indexOf("(os error 3)") === -1) {
          existsUnknown = true;
        }
      }

      var hwKeyGuard = hiwaterKey(filePath);

      // Echo bound (ultrareview on PR #224), shared by BOTH holds below: the
      // echo is MERGE MATERIAL, and merge material the model cannot hold is
      // not material — 64 KiB is a small local model's entire 16k-token
      // window, so anything bigger only burns the context these guards exist
      // to respect. Under the bound the full content ships and counts as the
      // read; over it, a head ships with a loud truncation notice (WHAT
      // dropped, WHY, disk unaffected) and the mark is NOT recorded, so the
      // next attempt bounces into an explicit ranged read_file rather than a
      // merge against a partial view.
      var ECHO_MAX = 65536;

      // P22 Tier 3: you cannot shrink what you have not seen. A shrinking
      // whole-file write over a file the session has never read, or whose
      // content CHANGED since it last read it, is a rewrite from a copy that
      // does not match the disk — the commonest way an assistant destroys a
      // file: it saw the file twenty messages ago, the history got compacted,
      // and it confidently rewrites from memory (2026-08-10 qwen: a fresh
      // step wrote 2449 bytes over a 5598-byte artifact with 22 checks
      // passing, zero reads in the step; the score fell to 1/42). Not a
      // refusal: the reply carries the file's CURRENT content to merge
      // against, counts itself as the read, and the next attempt proceeds.
      // One bounce, with the missing information.
      // Evaluated only once the cheap conditions hold, so an ordinary write
      // still costs no stat and no state read.
      var seenVerdict = (fileExists && !hiwaterExempt(hwKeyGuard) && bytes < existingSize &&
          typeof existing === "string") ? readSeenVerdict(filePath) : "seen";
      if (seenVerdict !== "seen") {
        // The reason clause, and ONLY the reason clause, differs between the
        // two verdicts — the way forward below is the same either way.
        var staleWhy = seenVerdict === "never"
          ? "but you have NEVER read this file in this session — you are rewriting it from a copy you never saw, "
          : "but the file has CHANGED since you last read it — your context copy is stale, ";
        if (existingSize <= ECHO_MAX) {
          readmarkPut(filePath);
          glog("write_file guard: stale-shrink echo for " + filePath + " (attempted " + bytes + " over " + existingSize + " bytes; read-mark verdict: " + seenVerdict + ")");
          return fail(
            "WRITE HELD — nothing was written and nothing is lost. You are shrinking " + filePath +
            " from " + existingSize + " to " + bytes + " bytes, " + staleWhy +
            "so this rewrite would silently destroy parts of the current file. " +
            "Here is the CURRENT content of " + filePath + ":\n\n" + existing +
            "\n\nMerge your change INTO this current text and call write_file again with the full merged content " +
            "(or use edit_file for a targeted change). This reply counts as your read — the file will not be held for this reason again."
          );
        }
        glog("write_file guard: stale-shrink hold (truncated echo) for " + filePath + " (attempted " + bytes + " over " + existingSize + " bytes; read-mark verdict: " + seenVerdict + ")");
        return fail(
          "WRITE HELD — nothing was written and nothing is lost. You are shrinking " + filePath +
          " from " + existingSize + " to " + bytes + " bytes, " + staleWhy +
          "so this rewrite would silently destroy parts of the current file. " +
          "The file is too large (" + existingSize + " bytes) to echo here; its first lines are:\n\n" + existing.substring(0, 4096) +
          "\n\n[TRUNCATED — only the first 4096 of " + existingSize + " bytes shown; the file on disk is complete and unaffected.] " +
          "This truncated preview does NOT count as reading the file. Call read_file(\"" + filePath + "\") " +
          "(with offset/limit for ranges) to see the current content, then merge your change into it — " +
          "after a real read this write will be accepted, or use edit_file for a targeted change."
        );
      }

      var hwBase = existingSize;
      var hwGoodBase = 0;
      if (fileExists && !hiwaterExempt(hwKeyGuard)) {
        var hwEntry = hiwaterEntryFor(hiwaterLoad(), filePath);
        var hwHi = hiwaterHi(hwEntry);
        // Monotone while the file exists: an out-of-band change since our
        // last write is folded in as evidence, never a reason to forget the
        // mass this file has held (the re-base rule was a laundering hole —
        // see the ratchet design comment above).
        if (hwHi > hwBase) {
          hwBase = hwHi;
        }
        hwGoodBase = hiwaterGood(hwEntry);
      }

      // Shrink floor, anchored on the last evidenced-good size when one
      // exists (see the ratchet design comment above).
      var floorBase = hwGoodBase > 0 ? hwGoodBase : hwBase;
      var floorAnchor = hwGoodBase > 0 ? "good" : "hi";
      if (!hiwaterExempt(hwKeyGuard) && floorBase > 500 && bytes < existingSize && bytes < floorBase * 0.3) {
        var sizeStory;
        if (floorAnchor === "good") {
          sizeStory = "holds " + existingSize + " bytes now, and its last version that passed a structural check held " + floorBase + " bytes";
        } else if (hwBase > existingSize) {
          sizeStory = "holds " + existingSize + " bytes now and has held " + hwBase + " bytes before";
        } else {
          sizeStory = "currently holds " + existingSize + " bytes";
        }
        glog("write_file guard: WRITE REFUSED (shrink floor) " + filePath + " attempted=" + bytes + " existing=" + existingSize + " floorBase=" + floorBase + " anchor=" + floorAnchor);
        return {
          content: "WRITE REFUSED — the file was NOT modified and is fully intact. " +
            "You tried to write only " + bytes + " bytes over " + filePath +
            " which " + sizeStory + " (" +
            Math.round(bytes / floorBase * 100) + "% of that). That usually means " +
            "you sent a fragment instead of the whole file. For a small change, use " +
            "edit_file instead: edit_file(file_path=\"" + filePath + "\", old_string=<the exact current text>, " +
            "new_string=<your replacement>) — it changes just that snippet and leaves the rest untouched. " +
            "To remove a section, edit_file with new_string=\"\". " +
            "Only if you truly mean to replace the WHOLE file: (1) read_file " + filePath + ", " +
            "(2) merge your change into the FULL text, (3) call write_file again with the complete content.",
          success: false
        };
      }

      // STRUCTURAL SHRINK HOLD (P23): bytes cannot see function removal.
      //
      // The byte floor answers "how much smaller?", which is the wrong
      // question for the destructive shape actually observed: a rewrite that
      // keeps the file's size respectable while dropping most of its
      // top-level definitions clears 30% comfortably and still deletes the
      // work. Two independent P23 analyses converged on the same guard — a
      // SHRINKING rewrite that removes more pre-existing definitions than it
      // keeps, or that drops definitions present in the last version that
      // PASSED a structural check, is held ONCE with the stale-shrink echo
      // treatment: the current content ships as merge material, the removed
      // definitions are NAMED, the echo counts as the read, and the removal
      // set is signed onto the ratchet entry. Send the same removal set again
      // and it goes through — the deletions were in view when it was
      // re-sent, which is the whole point of the bounce — and any successful
      // write clears the signature, so the next removal event re-arms it.
      //
      // Invariants kept: a NON-shrinking write is never refused; force is
      // never held; and the pass fails OPEN in every direction — either side
      // yielding zero parsed definitions, or the computation throwing, leaves
      // the write untouched. edit_file is exempt by design: an explicit
      // old_string IS the statement of intent this hold asks for.
      if (fileExists && !existsUnknown && !hiwaterExempt(hwKeyGuard) &&
          bytes < existingSize && typeof existing === "string") {
        var gutHold = null;
        try {
          preOldBodies = symbolBodies(existing);
          preNewBodies = symbolBodies(fileContent);
          var gutRemoved = [];
          var gutKept = 0;
          var gutOldCount = 0;
          for (var gOld in preOldBodies) {
            gutOldCount++;
            if (preNewBodies[gOld] === undefined) gutRemoved.push(gOld);
            else gutKept++;
          }
          var gutNewCount = 0;
          for (var gNew in preNewBodies) gutNewCount++;
          if (gutOldCount > 0 && gutNewCount > 0) {
            var gutEntry = hiwaterEntryFor(hiwaterLoad(), filePath);
            // The structural anchor: the definition set of the last version
            // that passed a structural check, recorded when `good` rebases.
            // Measured against the NEW content, not against disk — the point
            // of the anchor is that the current disk copy may itself already
            // have lost something the evidenced-good version had.
            var goodNames = gutEntry && Array.isArray(gutEntry.goodSyms)
              ? gutEntry.goodSyms : [];
            var lostFromGood = [];
            for (var gi = 0; gi < goodNames.length; gi++) {
              var gn = goodNames[gi];
              if (typeof gn !== "string") continue;
              if (preNewBodies[gn] !== undefined) continue;
              lostFromGood.push(gn);
            }
            if (gutRemoved.length > gutKept || lostFromGood.length > 0) {
              // One removal SET, deduplicated: the two arms overlap whenever
              // a definition is missing from both disk and the good anchor.
              var gutSet = gutRemoved.slice(0);
              for (var li = 0; li < lostFromGood.length; li++) {
                if (gutSet.indexOf(lostFromGood[li]) === -1) gutSet.push(lostFromGood[li]);
              }
              gutSet.sort();
              var gutSig = String(hashStr(gutSet.join("\n")));
              var gutAckd = gutEntry && typeof gutEntry.heldGut === "string" ? gutEntry.heldGut : "";
              if (gutAckd !== gutSig) {
                gutHold = { names: gutSet, sig: gutSig, fromGood: lostFromGood, old: gutOldCount, kept: gutKept };
              }
            }
          }
        } catch (eGut) {
          // No structural opinion — the write proceeds exactly as before.
          gutHold = null;
        }
        if (gutHold) {
          heldGutPut(filePath, gutHold.sig);
          var gutWhy = "This rewrite of " + filePath + " (" + existingSize + " → " + bytes +
            " bytes) REMOVES " + gutHold.names.length + " of the " + gutHold.old +
            " top-level sections the file defines and keeps " + gutHold.kept + ": " + nameList(gutHold.names) + "." +
            (gutHold.fromGood.length > 0
              ? " " + nameList(gutHold.fromGood) + " were present in the last version of this file that PASSED a structural check."
              : "") +
            " Deleting more than you keep is usually a rewrite from a stale or partial copy rather than a deliberate removal, and byte counts cannot see it.";
          var gutHow = "If those sections are genuinely obsolete, send this SAME content again and it will be written — " +
            "this hold fires once per removal set, not once per attempt. Otherwise put the missing sections back and " +
            "send the full merged file, or use edit_file to remove one named section at a time.";
          glog("write_file guard: structural shrink hold for " + filePath + " (" + existingSize + "->" + bytes +
            " bytes) removed=[" + gutHold.names.join(",") + "] kept=" + gutHold.kept + " sig=" + gutHold.sig);
          if (existingSize <= ECHO_MAX) {
            readmarkPut(filePath);
            return fail(
              "WRITE HELD — nothing was written and nothing is lost. " + gutWhy +
              "\n\nHere is the CURRENT content of " + filePath + ":\n\n" + existing +
              "\n\n" + gutHow + " This reply counts as your read."
            );
          }
          return fail(
            "WRITE HELD — nothing was written and nothing is lost. " + gutWhy +
            "\n\nThe file is too large (" + existingSize + " bytes) to echo here; its first lines are:\n\n" +
            existing.substring(0, 4096) +
            "\n\n[TRUNCATED — only the first 4096 of " + existingSize + " bytes shown; the file on disk is complete and unaffected.] " +
            "This truncated preview does NOT count as reading the file — call read_file(\"" + filePath +
            "\") (with offset/limit for ranges) to see what those sections contain. " + gutHow
          );
        }
      }
    }

    var pyGate = { ran: false, ok: false, detail: "" };
    if (!input.force) {
      // Versioned-copy REFUSAL (observed live: models fork foo.py.new2,
      // new_foo.py, foo_fixed_v1.txt instead of fixing the real file, then
      // lose track of which copy is real — an advisory did not stop it).
      var baseName = filePath.split("\\").join("/").split("/").pop().toLowerCase();
      var copyMarkers = [".new", "_v1", "_v2", "_v3", "_v4", "_v5", "_fixed", "_backup", "_temp", "_copy", "_part", "_old", "_final", "_clean", "_scrubbed", "scratch"];
      var copyPrefixes = ["new_", "copy_", "old_", "temp_", "backup_"];
      var copyHit = null;
      for (var m = 0; m < copyMarkers.length; m++) {
        if (baseName.indexOf(copyMarkers[m]) !== -1) { copyHit = copyMarkers[m]; break; }
      }
      if (!copyHit) {
        for (var p = 0; p < copyPrefixes.length; p++) {
          if (baseName.indexOf(copyPrefixes[p]) === 0) { copyHit = copyPrefixes[p]; break; }
        }
      }
      if (copyHit) {
        glog("write_file guard: WRITE REFUSED (versioned copy '" + copyHit + "') " + filePath);
        return fail("WRITE REFUSED — '" + filePath + "' looks like a versioned copy ('" + copyHit + "'). Nothing was written. Keep ONE real file: change the ORIGINAL in place with edit_file, or write the full corrected content directly to the original path (a complete valid rewrite at or above the file's current size is always accepted).");
      }

      // SUFFIXED-COPY REFUSAL: writing `minidb.sh` while `minidb` exists.
      //
      // The marker list above catches renames that ANNOUNCE themselves
      // (_v2, _backup, .new). This catches the quieter fork: keep the name,
      // add an extension. Observed live 2026-07-27 — a run built ./minidb to
      // 8/42 passing, then drifted onto ./minidb.sh and spent its remaining
      // time improving a file the acceptance tests never read.
      //
      // Deliberately narrow: it fires only when the target's STEM is itself
      // an existing file, i.e. the original carries no extension of its own.
      // That is the copy instinct. Sibling formats keep working, because
      // their stems are not files: config.json next to config.yaml, tool.ts
      // next to tool.js, index.css next to index.html.
      var lastDot = baseName.lastIndexOf(".");
      if (lastDot > 0) {
        var stem = baseName.substring(0, lastDot);
        var dirPart = filePath.split("\\").join("/");
        var slashAt = dirPart.lastIndexOf("/");
        var stemPath = slashAt >= 0 ? dirPart.substring(0, slashAt + 1) + stem : stem;
        // Nanna.stat THROWS when the path is absent, and its result carries
        // {size, is_file, is_dir, modified} — there is no `exists` flag. The
        // throw IS the existence check.
        var originalStat = null;
        try { originalStat = Nanna.stat(stemPath); } catch (e) { originalStat = null; }
        if (originalStat && originalStat.is_file) {
          return failEscalating(
            hiwaterKey(filePath),
            "WRITE REFUSED — '" + filePath + "' is '" + stem + "' with an extension added, and '" +
            stem + "' already exists (" + (originalStat.size || 0) + " bytes). Nothing was written. " +
            "That fork leaves two files and the real one stops improving — whatever reads '" + stem +
            "' (tests, callers) will not see this content. Work on '" + stemPath + "' itself: " +
            "edit_file(file_path=\"" + stemPath + "\", old_string=<exact current text>, new_string=<replacement>) " +
            "for a targeted change, or write the complete content to '" + stemPath + "'.",
            "STOP — '" + filePath + "' WILL NEVER BE ACCEPTED. You have tried it repeatedly. " +
            "The only file that counts is '" + stemPath + "'. Send this exact call instead:\n" +
            "    write_file(file_path=\"" + stemPath + "\", content=<the complete script>)\n" +
            "Do not add an extension. Do not try '" + filePath + "' again."
          );
        }
      }

      // VALID CONTENT ALWAYS WINS (round-13 lesson): the earlier rail
      // blocked whole-file writes whenever a parked draft existed — even
      // when the new content was perfectly valid, bouncing the model's one
      // reliable move (fresh generation) and wedging it between an empty
      // real file and a draft it would not repair. Order is now: check
      // validity FIRST; a parsing .py write is accepted outright and any
      // stale draft/markers are swept; only INVALID content meets the
      // park/rail machinery.
      pyGate = pythonSyntaxCheck(filePath, fileContent);
      var syntaxDetail = pyGate.ran && !pyGate.ok ? pyGate.detail : null;
      if (syntaxDetail === null) {
        var sweepBufPath = filePath + ".__buffer__";
        try {
          Nanna.exec("rm -f " + shq(sweepBufPath) + " " + shq(filePath + ".__cleared__") + " " + shq(sweepBufPath + ".__cleared__"), null, 15);
        } catch (eSweep) {
          // Stale draft leftovers are harmless.
        }
      } else {
        // Existing parked draft + ANOTHER invalid regeneration: keep the
        // parked draft authoritative and steer to the repair loop.
        var railBufPath = filePath + ".__buffer__";
        var railParked = null;
        try {
          railParked = Nanna.readFile(railBufPath);
        } catch (eRail) {
          // No parked draft.
        }
        if (railParked !== null && railParked !== undefined && railParked !== "") {
          glog("write_file guard: WRITE BLOCKED (invalid .py over parked draft) " + filePath);
          return fail("WRITE BLOCKED — this content has a SYNTAX ERROR (" + syntaxDetail + ") and a parked draft for " + filePath + " already exists at " + railBufPath + " (" + railParked.length + " chars). Repair THAT draft: edit_file(file_path=\"" + railBufPath + "\", old_string=<the broken line>, new_string=<the fix>), then file_buffer(action=\"commit\", file_path=\"" + filePath + "\"). A fully VALID rewrite of " + filePath + " at or above its current size would also be accepted.");
        }
      }

      // Draft PARKING (round-8 lesson): refusing a broken whole-file write
      // outright sends the model into a regeneration lottery — each retry
      // regenerates everything and rolls new errors (observed live: 21
      // refusals, every one a different line). Instead the rejected draft
      // is SAVED to the buffer beside the target, where the model repairs
      // the one named error with a small edit_file delta and commits.
      if (syntaxDetail) {
        var parkPath = filePath + ".__buffer__";
        var parked = false;
        try {
          Nanna.writeFile(parkPath, fileContent);
          parked = true;
        } catch (ePark) {
          // Fall through to the plain refusal below.
        }
        if (parked) {
          // Quote the offending line verbatim: it is a ready-made
          // old_string, removing the last excuse to regenerate (observed
          // live: three parks in a row, each a fresh full regeneration
          // with a brand-new error line).
          var lineQuote = "";
          if (syntaxDetail.indexOf("line ") === 0) {
            var colonAt = syntaxDetail.indexOf(":");
            if (colonAt > 5) {
              var lineNo = parseInt(syntaxDetail.substring(5, colonAt), 10);
              if (lineNo >= 1) {
                var draftLines = fileContent.split("\n");
                if (lineNo <= draftLines.length) {
                  var lq = draftLines[lineNo - 1];
                  if (lq.length > 80) lq = lq.substring(0, 80);
                  lineQuote = " Line " + lineNo + " of your draft is: `" + lq + "` — use exactly that as old_string.";
                }
              }
            }
          }
          glog("write_file guard: WRITE PARKED (invalid .py) " + filePath + " draft=" + bytes + " bytes");
          return fail("WRITE PARKED — your content for " + filePath + " has a SYNTAX ERROR (" + syntaxDetail + "), so the file was NOT changed. Nothing was lost: the draft IS SAVED at " + parkPath + "." + lineQuote + " Your NEXT call must be edit_file(file_path=\"" + parkPath + "\", old_string=<the broken line>, new_string=<the fixed line>), then file_buffer(action=\"commit\", file_path=\"" + filePath + "\"). Do NOT call write_file again for this file and do NOT regenerate it.");
        }
        return fail("WRITE REFUSED — the content you sent for " + filePath + " is NOT valid Python (" + syntaxDetail + "). The file is UNCHANGED. Fix the syntax and call write_file again with the corrected COMPLETE text.");
      }
    }

    // P22 Tier 3: displaced content stays recoverable. A full overwrite
    // parks the outgoing version at <file>.__prev__ (one slot, overwritten
    // each time) so a destructive rewrite is one read_file away from
    // recovery instead of gone — three bench legs ended with their peak
    // artifact overwritten and nothing to restore. Best-effort, never
    // blocks the write. Skipped under force (the pre-image was never read;
    // force is an explicit re-shape) and for managed sidecars.
    var prevPath = filePath + ".__prev__";
    var prevParked = false;
    if (fileExists && !input.force && typeof existing === "string" &&
        existing !== fileContent && !hiwaterExempt(hiwaterKey(filePath))) {
      try {
        Nanna.writeFile(prevPath, existing);
        prevParked = true;
      } catch (ePrev) {
        // Recovery copy is best-effort.
      }
    }

    try {
      Nanna.writeFile(filePath, fileContent);
    } catch (e2) {
      // Classify BEFORE truncating (P23): the cause is the tail, and the
      // advice that follows it must match the cause — a permission denial is
      // not something a retry can fix.
      return fail("write_file failed writing " + filePath + " " +
        writeFailureNote(filePath, String(e2),
          "Retry the same call; if it fails again, read_file to verify the file state."));
    }

    // Structural verdict on what landed. The .py gate above already ran the
    // real check — adopt its answer instead of spawning the interpreter a
    // second time; under force (gate skipped) the generic check runs.
    var hwKeyPost = hiwaterKey(filePath);
    var verdict = null;
    var prevChk = null;
    var escNote = "";
    if (!hiwaterExempt(hwKeyPost)) {
      try {
        var peek = hiwaterEntryFor(hiwaterLoad(), filePath);
        if (peek && (peek.chk === "ok" || peek.chk === "bad")) prevChk = peek.chk;
      } catch (ePeek) {
        // No history — the sentence just has no before/after clause.
      }
      var checkKind = structuralCheckKind(filePath, fileContent);
      if (checkKind === "py" && pyGate.ran) {
        verdict = { ok: pyGate.ok, tool: "python ast", detail: pyGate.detail };
      } else if (checkKind) {
        verdict = runStructuralCheck(checkKind, filePath, fileContent);
      }
      // force writes bytes through untouched, so it inherits the escape
      // repair's exemption here too.
      if (!input.force) escNote = escapedCommentNote(checkKind, fileContent);
    }
    if (input.force || !fileExists) prevChk = null;

    // Ratchet update AFTER a successful write. While the file exists, the
    // high-water mark only ever RISES — max of the previous mark, the disk
    // size we just observed, and this write — so fluctuating rewrites stay
    // pinned to the peak and out-of-band changes fold in as evidence
    // instead of restarting the history (the old re-base-on-mismatch rule
    // was the laundering hole documented in the ratchet design comment).
    // Force and creation-over-missing RESET it — both are deliberate
    // re-shapes with nothing stale worth protecting, and that includes the
    // evidenced-good anchor. A passing structural check records this
    // write's size as the new good anchor. When the pre-write read failed
    // for unknown reasons, the state is left completely alone.
    if (!existsUnknown) {
      try {
        var hwMap = hiwaterLoad();
        var hwKey = hiwaterKey(filePath);
        if (!hiwaterExempt(hwKey)) {
          var hwPrevEntry = hiwaterEntryFor(hwMap, filePath);
          var hwNext = bytes > existingSize ? bytes : existingSize;
          if (!input.force && fileExists) {
            var hwPrevHi = hiwaterHi(hwPrevEntry);
            if (hwPrevHi > hwNext) hwNext = hwPrevHi;
          }
          if (input.force || !fileExists) hwNext = bytes;
          // The entry is rebuilt from scratch every time, which is also how
          // the structural hold's `heldGut` signature clears itself: any
          // successful write drops it and the hold re-arms for the next
          // removal event.
          var hwNew = { hi: hwNext, last: bytes, at: Date.now() };
          if (!input.force && fileExists && hwPrevEntry) {
            if (hiwaterGood(hwPrevEntry) > 0) {
              hwNew.good = hwPrevEntry.good;
              hwNew.goodAt = hwPrevEntry.goodAt || 0;
              if (Array.isArray(hwPrevEntry.goodSyms)) {
                hwNew.goodSyms = hwPrevEntry.goodSyms;
              }
            }
            if (hwPrevEntry.chk === "ok" || hwPrevEntry.chk === "bad") hwNew.chk = hwPrevEntry.chk;
            // Coverage high-water record travels with the ratchet: force and
            // creation-over-missing reset it exactly as they reset `good`.
            if (typeof hwPrevEntry.bestSyms === "number" && isFinite(hwPrevEntry.bestSyms)) {
              hwNew.bestSyms = hwPrevEntry.bestSyms;
              hwNew.bestAt = hwPrevEntry.bestAt || 0;
            }
          }
          if (verdict) {
            hwNew.chk = verdict.ok ? "ok" : "bad";
            if (verdict.ok) {
              hwNew.good = bytes;
              hwNew.goodAt = Date.now();
              // The structural anchor rebases WITH the good size: the symbol
              // table for this content is already computed when the hold ran,
              // so recording it costs at most one parse on the paths that
              // never needed one (and the rewrite-note below then reuses it).
              // A stale set would name definitions the good version no longer
              // has, so it is always replaced — and DROPPED if the pass
              // cannot run at all — never carried past a rebase.
              try {
                if (!preNewBodies) preNewBodies = symbolBodies(fileContent);
                hwNew.goodSyms = symbolNames(fileContent, preNewBodies);
              } catch (eSyms) {
                delete hwNew.goodSyms;
              }
            }
          }
          hwMap[hwKey] = hwNew;
          hiwaterSave(hwMap);
        }
      } catch (eHw) {
        // Best-effort; the user's write already succeeded.
      }
      // The file now holds exactly what the model sent — its knowledge of
      // the content is perfect at this instant, so the write counts as a
      // read for the stale-shrink guard.
      if (!hiwaterExempt(hwKeyPost)) readmarkPut(filePath);
    }

    var structNote = structSentence(filePath, verdict, prevChk) + escNote;
    if (verdict && !verdict.ok) {
      glog("write_file structure: " + filePath + " does NOT parse after write (" + verdict.tool + "): " + verdict.detail);
    } else if (verdict && verdict.ok && prevChk === "bad") {
      glog("write_file structure: " + filePath + " parses again after write (" + verdict.tool + ")");
    }

    // Rewrite-delta announcement (P22 Tier 3, bidirectional). The original
    // 2026-08-08 ornith lesson: full-file rewrites from a stale in-context
    // copy silently deleted working sections all run long — every write
    // reported plain success while the artifact kept 8 of 37 verified
    // commands. The 2026-08-10 follow-up showed the symmetric hole: BOTH
    // destructive writes that leg GREW the file (3333→6986 bytes, a
    // superset of function names with rewritten bodies) and the removal-
    // only note was structurally blind to them. The note now announces
    // (a) sections REMOVED, (b) pre-existing sections whose BODIES changed
    // (cheap per-symbol content hash), and (c) a rewrite that more than
    // doubled the file — i.e. added more new mass than the whole file
    // previously held. Refusing is still wrong (a complete rewrite is the
    // model's one reliable move); the note informs, is logged at INFO, and
    // names the .__prev__ recovery copy. Detection is a single whole-string
    // regex pass per side (no per-line split — the Boa split cost lesson):
    // shell functions `name() {`, case arms `name)`, and def/class/function
    // declarations at line starts. Informational only; never blocks.
    function topSymbolSpans(text) {
      var out = [];
      var re = /^[ \t]*(?:([A-Za-z_][A-Za-z0-9_]*)[ \t]*\(\)[ \t]*\{?[ \t]*$|(?:def|class|function)[ \t]+([A-Za-z_$][A-Za-z0-9_$]*)|([A-Za-z_][A-Za-z0-9_-]*)\)[ \t]*$)/gm;
      var m;
      while ((m = re.exec(text)) !== null) {
        var n = m[1] || m[2] || m[3];
        if (n) out.push({ name: n, at: m.index });
        if (m.index === re.lastIndex) re.lastIndex++;
      }
      return out;
    }
    function collapseWs(s) {
      var out = "";
      var pend = false;
      for (var ci = 0; ci < s.length; ci++) {
        var c = s.charAt(ci);
        if (c === " " || c === "\t" || c === "\r" || c === "\n") { pend = true; continue; }
        if (pend && out !== "") out += " ";
        pend = false;
        out += c;
      }
      return out;
    }
    function hashStr(s) {
      var h = 5381;
      for (var hi2 = 0; hi2 < s.length; hi2++) h = (((h << 5) + h) ^ s.charCodeAt(hi2)) >>> 0;
      return h;
    }
    // Where a symbol's body ends. The span is bounded by the next
    // declaration, then trimmed back past trailing TOP-LEVEL code — a bare
    // `usage` call after the closing brace must not leak into the hash, or
    // appending anything after the LAST function marks that function
    // "changed": a false positive that teaches the model to ignore the
    // note (caught by the P22 harness). A brace declaration runs through
    // its first column-0 closer regardless of body indentation (small
    // models write unindented shell); anything else (def/class, case arms)
    // follows the indentation rule.
    function symbolBodyEnd(text, declStart, limit) {
      var nl = text.indexOf("\n", declStart);
      var declEnd = (nl === -1 || nl > limit) ? limit : nl;
      var braced = text.substring(declStart, declEnd).indexOf("{") !== -1;
      var end = declEnd;
      var lineStart = declEnd + 1;
      while (lineStart < limit) {
        nl = text.indexOf("\n", lineStart);
        var lineEnd = (nl === -1 || nl > limit) ? limit : nl;
        var raw = text.substring(lineStart, lineEnd);
        var ti = 0;
        while (ti < raw.length && (raw.charAt(ti) === " " || raw.charAt(ti) === "\t")) ti++;
        var trimmed = raw.substring(ti);
        if (trimmed.charAt(trimmed.length - 1) === "\r") trimmed = trimmed.substring(0, trimmed.length - 1);
        var isCloser = trimmed === "}" || trimmed === "};" || trimmed === "fi" ||
          trimmed === "esac" || trimmed === "done" || trimmed === "end" || trimmed === ";;";
        if (braced) {
          end = lineEnd;
          if (ti === 0 && isCloser) break;
        } else {
          if (trimmed === "" || ti > 0 || isCloser) { end = lineEnd; }
          else break;
        }
        if (nl === -1 || nl >= limit) break;
        lineStart = nl + 1;
      }
      return end;
    }
    // Each symbol's body hash (whitespace-insensitive). First occurrence of
    // a duplicated name wins.
    function symbolBodies(text) {
      var spans = topSymbolSpans(text);
      var map = {};
      for (var si = 0; si < spans.length; si++) {
        if (map[spans[si].name] !== undefined) continue;
        var limit = si + 1 < spans.length ? spans[si + 1].at : text.length;
        var end = symbolBodyEnd(text, spans[si].at, limit);
        map[spans[si].name] = hashStr(collapseWs(text.substring(spans[si].at, end)));
      }
      return map;
    }
    // The sorted definition NAMES of a version, reusing an already-computed
    // body map when the caller has one (the structural hold's pre-write pass
    // is over exactly this content) so no side is parsed twice.
    function symbolNames(text, cachedBodies) {
      var bodies = cachedBodies || symbolBodies(text);
      var names = [];
      for (var n in bodies) names.push(n);
      names.sort();
      return names;
    }
    function nameList(arr) {
      var shown = arr.slice(0, 10);
      var more = arr.length - shown.length;
      return "`" + shown.join("`, `") + "`" + (more > 0 ? " (+" + more + " more)" : "");
    }
    var lossNote = "";
    var bestPath = filePath + ".__best__";
    if (fileExists && !input.force && typeof existing === "string" && existing !== fileContent) {
      try {
        var oldBodies = preOldBodies || symbolBodies(existing);
        var newBodies = preNewBodies || symbolBodies(fileContent);
        var dropped = [];
        var changed = [];
        for (var sName in oldBodies) {
          if (newBodies[sName] === undefined) dropped.push(sName);
          else if (newBodies[sName] !== oldBodies[sName]) changed.push(sName);
        }
        // COVERAGE HIGH-WATER PARK (P23). .__prev__ holds exactly ONE
        // generation, so a two-write spiral rotates the good version out: the
        // second write parks the wreck the first one made and the peak is
        // gone. .__best__ parks the outgoing version whenever it is the
        // richest version this ledger has seen, measured by top-level
        // definition COUNT — deliberately NOT a subset relation, because
        // rewrite spirals RENAME symbols, so no mid-spiral wreck is ever a
        // strict subset and the peak would rotate out on the second write
        // anyway. Only writes that actually REMOVE sections qualify (an
        // additive write has nothing to preserve), the first qualifying write
        // parks unconditionally, and the record resets with the ratchet on
        // force and creation. Best-effort; never blocks the write.
        var bestNote = "";
        if (dropped.length > 0 && !hiwaterExempt(hwKeyPost)) {
          var outCount = 0;
          for (var oName in oldBodies) outCount++;
          try {
            var bMap = hiwaterLoad();
            var bEntry = hiwaterEntryFor(bMap, filePath);
            var bHave = bEntry && typeof bEntry.bestSyms === "number" && isFinite(bEntry.bestSyms)
              ? bEntry.bestSyms : 0;
            if (outCount > bHave) {
              Nanna.writeFile(bestPath, existing);
              if (!bEntry) bEntry = { hi: existingSize, last: bytes, at: Date.now() };
              bEntry.bestSyms = outCount;
              bEntry.bestAt = Date.now();
              bMap[hwKeyPost] = bEntry;
              hiwaterSave(bMap);
              glog("write_file parked coverage high-water for " + filePath + " at " + bestPath +
                " (" + outCount + " sections, previous record " + bHave + ")");
            } else if (bHave > 0) {
              // Parked best differs from the version .__prev__ just took:
              // name it, with the counts that make the difference concrete.
              bestNote = " The fullest prior version (" + bHave + " sections, against " + outCount +
                " in the version this write replaced) is parked at " + bestPath +
                " — read_file it before rewriting again.";
            }
          } catch (eBest) {
            // Recovery copies are best-effort.
          }
        }
        // "More than doubled" = this rewrite added more new mass than the
        // entire previous file held — a from-scratch superset, the shape of
        // both destructive ornith writes. The 500-byte gate matches the
        // ratchet's own smallness line.
        var grewPastDouble = existingSize > 500 && bytes > existingSize * 2;
        if (dropped.length > 0 || changed.length > 0 || grewPastDouble) {
          var parts = "";
          if (dropped.length > 0) {
            parts += " It REMOVED sections that existed on disk before: " + nameList(dropped) + ".";
          }
          if (changed.length > 0) {
            parts += " It CHANGED the bodies of pre-existing sections: " + nameList(changed) + ".";
          }
          if (grewPastDouble) {
            parts += dropped.length === 0 && changed.length === 0
              ? " It more than doubled the file while keeping every pre-existing section's body intact — the growth is purely additive."
              : " It also more than doubled the file — most of what is on disk now is new text.";
          }
          lossNote = " NOTE: this whole-file rewrite replaced " + filePath + " (" + existingSize + " → " + bytes + " bytes)." + parts +
            " The write SUCCEEDED — the file holds exactly what you sent. If any of those sections were verified working, re-verify them now" +
            (prevParked ? "; the previous version is preserved at " + prevPath + " (read_file it to recover anything)" : "") + "." + bestNote;
          glog("write_file rewrite-note for " + filePath + " (" + existingSize + "->" + bytes + " bytes): removed=[" + dropped.join(",") + "] changed=[" + changed.join(",") + "]" + (grewPastDouble ? " grew>2x" : ""));
        }
      } catch (eLoss) {
        // Informational only — never affects the write.
      }
    }

    // Deliberately NO echo of the written content: echoing the whole file
    // made the result exceed the context threshold, and the model read the
    // resulting truncation stub as "my write was discarded" (observed
    // live, round 7). The file on disk is the source of truth.
    // A repair the model cannot see is a repair it will "fix" back. Announce
    // WHAT changed, WHY, and that the write SUCCEEDED — an unexplained
    // difference between what was sent and what landed reads as corruption.
    var repairNote = "";
    if (unescapedNewlines > 0) {
      repairNote =
        " NOTE: your content arrived with " + unescapedNewlines +
        " literal backslash-n sequences and no real line breaks, which would have made the file one unusable line;" +
        " they were converted to real newlines before writing. The write SUCCEEDED and the file is correct —" +
        " nothing to redo. Send real line breaks next time and this note goes away.";
    }
    // Carry the structural OUTCOME, the same way edit_file does and for the
    // same reason: this tool has already run the file's real parser, and a
    // consumer reading only "the bytes landed" records a break as landed work.
    // Set only when a checker actually ran — an absent or fail-open verdict
    // must never read as broken.
    var result = { content: "Wrote " + bytes + " bytes to " + filePath + ". The file on disk now holds exactly this content." + structNote + repairNote + lossNote };
    if (verdict) {
      result.data = { structure: { parses: verdict.ok === true, tool: verdict.tool, detail: verdict.detail } };
    }
    return result;
  }
}

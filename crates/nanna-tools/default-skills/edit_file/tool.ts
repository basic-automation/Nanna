export default {
  name: "edit_file",
  version: "0.1.12",
  output: "memory",
  description: "Replace one exact text snippet in a file with new text — an in-place edit for small changes. Use this instead of rewriting the whole file with write_file. ALL THREE main parameters are REQUIRED: file_path, old_string, new_string. old_string must be text that exists in the file (copy it verbatim; indentation differences are tolerated) — include 2-3 surrounding lines to make it unique. Only the matched snippet changes; the rest of the file is untouched. After each edit the cheapest structural check (sh -n / node --check / JSON.parse) runs on the result and its verdict is appended — including whether the file parsed before the edit. Use write_file only for new files or full rewrites.",
  parameters: {
    type: "object",
    properties: {
      file_path: { type: "string", description: "REQUIRED. Path to the file to edit. Relative paths are resolved against the workspace directory." },
      old_string: { type: "string", description: "REQUIRED. The exact text currently in the file to be replaced. Copy it verbatim (read_file first if unsure) and include 2-3 surrounding lines so it matches exactly once." },
      new_string: { type: "string", description: "REQUIRED. The replacement text. May be empty to delete the snippet. Must differ from old_string." },
      replace_all: { type: "boolean", description: "If true, replace EVERY occurrence of old_string. Default: false." },
      occurrence: { type: "integer", description: "Replace only the Nth match of old_string, 1-based. Alternative to replace_all when old_string appears more than once." }
    },
    required: ["file_path", "old_string", "new_string"]
  },
  execute: function(input) {
    // Errors are returned as { content, success: false } instead of thrown:
    // a thrown script error reaches the model wrapped in five stacked
    // "Execution failed:" prefixes, which small models read as corruption
    // and spiral on. A structured failure surfaces as clean corrective text.
    function fail(message) {
      return { content: message, success: false };
    }

    // Guard events are logged at INFO so they can be audited from the
    // daemon log after a run (P22: a guard whose firing leaves no log line
    // is indistinguishable from a guard that never fired). Best-effort.
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

    // Repeat-refusal escalation, shared with write_file — see the long note
    // there. Same state file and the same "fork:" key prefix, so attempts
    // alternating between the two tools accumulate on ONE counter instead of
    // each tool granting the model a fresh allowance. Fails open throughout.
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
    // read_file's ONLY output format is "<right-aligned line number><TAB><line>",
    // and the miss message below tells the model to copy that text back as
    // old_string — so the product points at text that can never match. Undo the
    // format, but only on read_file's exact shape: EVERY non-empty line
    // prefixed, numbers consecutive, all right-aligned to one common width.
    // A bare per-line number+tab test is unsafe because tab-separated data
    // files start that way, and stripping their first column would silently
    // edit the wrong text.
    function stripLineNumberBlock(text) {
      var lines = text.split("\r\n").join("\n").split("\n");
      var width = -1;
      var prev = -1;
      var out = [];
      var seen = 0;
      for (var i = 0; i < lines.length; i++) {
        if (lines[i] === "") { out.push(""); continue; }
        var tab = lines[i].indexOf("\t");
        if (tab < 1) return null;
        var numField = lines[i].substring(0, tab);
        if (!/^ *\d+$/.test(numField)) return null;
        if (width === -1) width = numField.length;
        else if (numField.length !== width) return null;
        var n = parseInt(numField, 10);
        if (prev !== -1 && n !== prev + 1) return null;
        prev = n;
        seen++;
        out.push(lines[i].substring(tab + 1));
      }
      // One numbered line is a coincidence; a block is a paste.
      if (seen < 2) return null;
      return out.join("\n");
    }

    function failEscalating(pathKey, normal, blunt) {
      var n = refusalBump("fork:" + pathKey);
      glog("edit_file guard refused (fork, attempt " + n + "): " + pathKey);
      return fail(n > REFUSAL_ESCALATE_AT ? blunt : normal);
    }

    // Anti-erosion ratchet state, shared with write_file v0.1.15 /
    // file_buffer (the design comment lives in write_file). All state I/O
    // is best-effort and fails OPEN — it can never block an edit.
    var HIWATER_STATE = ".nanna/write_hiwater.json";
    var HIWATER_MAX_ENTRIES = 200;
    function hiwaterNormKey(path) {
      var k = path.split("\\").join("/");
      // Case folds only where the filesystem does: on Linux `README.md` and
      // `readme.md` are two files, and one key let a read of either count for both.
      if (Nanna.platform === "win32" || Nanna.platform === "darwin") k = k.toLowerCase();
      while (k.indexOf("./") === 0) k = k.substring(2);
      while (k.indexOf("//") !== -1) k = k.split("//").join("/");
      return k;
    }
    // Canonical key (P22): absolute spellings under the workspace root
    // collapse to the relative form so one file has ONE ledger entry — the
    // observed split-brain ('minidb' hi=9768 next to 'd:/.../minidb'
    // hi=3195) let an absolute-path edit cut the effective floor by two
    // thirds. Legacy spellings are folded forward by hiwaterEntryFor.
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
    function hiwaterIsBuffer(key) {
      var buf = ".__buffer__";
      return key.length >= buf.length && key.lastIndexOf(buf) === key.length - buf.length;
    }
    function hiwaterIsPrev(key) {
      var p = ".__prev__";
      return key.length >= p.length && key.lastIndexOf(p) === key.length - p.length;
    }
    // The coverage high-water park write_file maintains — a recovery copy
    // like .__prev__, rewritten wholesale, never judged by history.
    function hiwaterIsBest(key) {
      var b = ".__best__";
      return key.length >= b.length && key.lastIndexOf(b) === key.length - b.length;
    }
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
    // Canonical + legacy spellings folded into one entry (hi = max, the
    // rest follows the fresher entry); the alias is dropped from the map so
    // the split-brain heals on the next save through it.
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
    // The shrink-floor base for a path, or 0 when there is no trustworthy
    // one. Anchored on the last EVIDENCED-GOOD size — the newest version
    // that passed a structural check — falling back to the monotone
    // high-water mark when no check has ever passed (P22: the largest-ever
    // anchor protected a 9768-byte draft that scored 2/42 and nearly
    // refused the leg's eventual peak; size is not quality). The full
    // design comment lives in write_file.
    function hiwaterFloorInfo(path) {
      try {
        var key = hiwaterKey(path);
        if (hiwaterExempt(key)) return { base: 0, anchor: "" };
        var entry = hiwaterEntryFor(hiwaterLoad(), path);
        if (!entry) return { base: 0, anchor: "" };
        var good = hiwaterGood(entry);
        if (good > 0) return { base: good, anchor: "good" };
        var hi = hiwaterHi(entry);
        if (hi > 0) return { base: hi, anchor: "hi" };
        return { base: 0, anchor: "" };
      } catch (e) {
        return { base: 0, anchor: "" };
      }
    }

    function hiwaterRecord(path, newSize, prevSize, verdict, goodSyms) {
      try {
        var key = hiwaterKey(path);
        if (hiwaterExempt(key)) return;
        var map = hiwaterLoad();
        var entry = hiwaterEntryFor(map, path);
        var hi = newSize > prevSize ? newSize : prevSize;
        // Monotone: the previous mark survives regardless of who touched
        // the file in between.
        if (entry && typeof entry.hi === "number" && isFinite(entry.hi) && entry.hi > hi) {
          hi = entry.hi;
        }
        // Rebuilt from scratch, which is also how write_file's structural
        // shrink hold clears its `heldGut` signature: any successful
        // mutation drops it and the hold re-arms for the next removal event.
        var next = { hi: hi, last: newSize, at: Date.now() };
        if (entry) {
          if (hiwaterGood(entry) > 0) {
            next.good = entry.good;
            next.goodAt = entry.goodAt || 0;
            if (Array.isArray(entry.goodSyms)) {
              next.goodSyms = entry.goodSyms;
            }
          }
          if (entry.chk === "ok" || entry.chk === "bad") next.chk = entry.chk;
          // write_file's coverage high-water record survives in-band edits.
          if (typeof entry.bestSyms === "number" && isFinite(entry.bestSyms)) {
            next.bestSyms = entry.bestSyms;
            next.bestAt = entry.bestAt || 0;
          }
        }
        // A structural verdict updates the evidence: a pass records this
        // size as the new good anchor, either outcome records the state the
        // file was left in (the before/after clause of the next check).
        if (verdict) {
          next.chk = verdict.ok ? "ok" : "bad";
          if (verdict.ok) {
            next.good = newSize;
            next.goodAt = Date.now();
            // The anchor's definition set rebases WITH its size (P23): the
            // structural shrink hold in write_file measures removals against
            // it, and a set left over from an older good version would name
            // definitions this edit legitimately removed.
            //
            // But it must not rebase DOWNWARD on a shrinking set. A parsing
            // edit that drops names used to overwrite the anchor with the
            // smaller set, erasing the very evidence the write-side hold is
            // built on — so an edit could delete definitions and then a later
            // whole-file rewrite measured itself against the already-reduced
            // anchor and passed. Names the anchor still holds are kept; new
            // ones are added.
            if (Array.isArray(goodSyms)) {
              var priorSyms = Array.isArray(next.goodSyms) ? next.goodSyms : [];
              var merged = goodSyms.slice();
              var lost = [];
              for (var gi = 0; gi < priorSyms.length; gi++) {
                if (goodSyms.indexOf(priorSyms[gi]) === -1) {
                  merged.push(priorSyms[gi]);
                  lost.push(priorSyms[gi]);
                }
              }
              if (lost.length > 0) {
                glog("edit_file anchor: keeping " + lost.length + " definition(s) the edit removed in the shrink anchor for " + path + ": [" + lost.join(",") + "]");
              }
              next.goodSyms = merged;
            } else {
              delete next.goodSyms;
            }
          }
        }
        map[key] = next;
        hiwaterSave(map);
      } catch (e) {
        // Best-effort.
      }
    }
    // Top-level definition names of a version — the same single-regex pass
    // write_file's rewrite-note uses (shell functions `name() {`, case arms
    // `name)`, def/class/function declarations at line starts), reduced to
    // sorted unique names because only membership matters here. One whole-
    // string regex pass, no per-line split (the Boa split-cost lesson).
    function symbolNames(text) {
      var re = /^[ \t]*(?:([A-Za-z_][A-Za-z0-9_]*)[ \t]*\(\)[ \t]*\{?[ \t]*$|(?:def|class|function)[ \t]+([A-Za-z_$][A-Za-z0-9_$]*)|([A-Za-z_][A-Za-z0-9_-]*)\)[ \t]*$)/gm;
      var seen = {};
      var names = [];
      var m;
      while ((m = re.exec(text)) !== null) {
        var n = m[1] || m[2] || m[3];
        if (n && seen[n] === undefined) {
          seen[n] = 1;
          names.push(n);
        }
        if (m.index === re.lastIndex) re.lastIndex++;
      }
      names.sort();
      return names;
    }

    // USER-DECLARED FILE INVARIANTS (P23), shared contract with write_file /
    // file_buffer (full design comment in write_file). Durable prohibitions
    // the USER stated in chat are registered at plan time and consulted
    // before any mutation; the refusal quotes the user's own sentence back.
    // Missing, unreadable or malformed registry => NO invariants (fail open,
    // silently), and force does NOT bypass — only the user lifts a
    // constraint, and ask_user is the wanted escape hatch.
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
    // `**` crosses directory separators, `*`/`?` stop at one; a wildcard-free
    // glob matches the path itself or anything under it. Matching runs over
    // the same canonical spelling the ratchet ledger uses. Fails open.
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
          // `no_delete` is about deletion and is enforced where deletions
          // happen; an edit is neither a creation nor a deletion, so
          // `no_create_under` cannot bite here either — edit_file only ever
          // changes a file that already exists.
          if (kind !== "read_only") continue;
          if (typeof inv.glob !== "string") continue;
          if (!invariantMatches(inv.glob, canon, norm)) continue;
          var quoted = typeof inv.source === "string" && inv.source !== "" ? inv.source : "";
          var scope = typeof inv.scope === "string" && inv.scope !== "" ? inv.scope : "session";
          glog("edit_file guard: " + verb + " REFUSED (declared invariant " + kind + " on '" + inv.glob + "') " + path);
          return verb + " REFUSED — " + path + " is under a path you declared off-limits" +
            (quoted === "" ? " (" + kind + " on `" + inv.glob + "`)" : ": \"" + quoted + "\"") +
            ". Nothing was changed and the file on disk is intact. That is YOUR instruction (declared for " +
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

    // WRITE-FAILURE HONESTY (P23), shared with write_file / file_buffer (full
    // design comment in write_file): classify BEFORE truncating, always keep
    // the trailing cause, and only offer a retry for a plausibly transient
    // fault. Parses both the current bridge format and one carrying the
    // stable ErrorKind name, falling back to the locale-independent OS error
    // number.
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
    // The same 120-char identification width as before, split so the trailing
    // cause survives the cut.
    function preserveCause(msg) {
      if (msg.length <= 120) return msg;
      return msg.substring(0, 80) + " … " + msg.substring(msg.length - 40);
    }
    function writeFailureNote(path, rawErr, retryAdvice) {
      var kind = writeErrorKind(rawErr);
      var shown = preserveCause(rawErr);
      if (kind === "PermissionDenied") {
        // FileStat carries no read-only bit, so the stat distinguishes what
        // it CAN and the sentence claims only the denial the OS reported.
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

    // Collapse whitespace runs in one line: leading/trailing dropped,
    // internal runs become a single space. The unit of indentation-tolerant
    // comparison.
    function normLine(line) {
      var out = "";
      var pendingWs = false;
      for (var i = 0; i < line.length; i++) {
        var c = line.charAt(i);
        if (c === " " || c === "\t" || c === "\r") {
          pendingWs = true;
          continue;
        }
        if (pendingWs && out !== "") out += " ";
        pendingWs = false;
        out += c;
      }
      return out;
    }

    // Dice bigram similarity of two normalized lines (0..1). Cheap, no
    // regex, good enough to point the model at the right neighborhood.
    function diceSim(a, b) {
      if (a === b) return 1;
      if (a.length < 2 || b.length < 2) return 0;
      var counts = {};
      var i;
      for (i = 0; i < a.length - 1; i++) {
        var bg = "k" + a.substring(i, i + 2);
        counts[bg] = (counts[bg] || 0) + 1;
      }
      var hits = 0;
      for (i = 0; i < b.length - 1; i++) {
        var bg2 = "k" + b.substring(i, i + 2);
        if (counts[bg2] > 0) { counts[bg2]--; hits++; }
      }
      return (2 * hits) / (a.length - 1 + b.length - 1);
    }

    // The file line most similar to old_string's first substantial line,
    // quoted with up to 3 following lines — a re-anchoring gift: every
    // failed match hands the model REAL text to copy into its next call.
    function closestSnippet(content, oldStr) {
      var target = "";
      var oldLines = oldStr.split("\n");
      for (var i = 0; i < oldLines.length; i++) {
        var t = normLine(oldLines[i]);
        if (t !== "") { target = t; break; }
      }
      if (target === "") return "";
      var lines = content.split("\n");
      var scan = lines.length < 500 ? lines.length : 500;
      var bestIdx = -1;
      var bestScore = 0.3; // below this it's noise, not an anchor
      for (var j = 0; j < scan; j++) {
        var s = diceSim(normLine(lines[j]), target);
        if (s > bestScore) { bestScore = s; bestIdx = j; }
      }
      if (bestIdx < 0) return "";
      var out = [];
      for (var k = bestIdx; k < lines.length && k < bestIdx + 4; k++) out.push(lines[k]);
      var snip = out.join("\n");
      if (snip.length > 240) snip = snip.substring(0, 240);
      return snip;
    }

    // Whitespace-tolerant match: find spans of whole file lines whose
    // normalized forms equal old_string's normalized lines. Replaces the
    // exact ORIGINAL span, so surrounding bytes (and their line endings)
    // are untouched. Observed live: a 9B model's old_string is composed
    // from compressed memory — content right, indentation wrong.
    function findLooseSpans(content, oldStr) {
      var oldLines = oldStr.split("\n");
      while (oldLines.length > 0 && normLine(oldLines[0]) === "") oldLines.shift();
      while (oldLines.length > 0 && normLine(oldLines[oldLines.length - 1]) === "") oldLines.pop();
      if (oldLines.length === 0) return [];
      var normOld = [];
      for (var i = 0; i < oldLines.length; i++) normOld.push(normLine(oldLines[i]));

      var starts = [0];
      for (var p = 0; p < content.length; p++) {
        if (content.charAt(p) === "\n") starts.push(p + 1);
      }
      var spans = [];
      for (var li = 0; li + normOld.length <= starts.length; li++) {
        var okAll = true;
        for (var lj = 0; lj < normOld.length; lj++) {
          var ls = starts[li + lj];
          var le = (li + lj + 1 < starts.length) ? starts[li + lj + 1] - 1 : content.length;
          if (normLine(content.substring(ls, le)) !== normOld[lj]) { okAll = false; break; }
        }
        if (okAll) {
          var endLine = li + normOld.length - 1;
          var spanEnd = (endLine + 1 < starts.length) ? starts[endLine + 1] - 1 : content.length;
          if (spanEnd > starts[li] && content.charAt(spanEnd - 1) === "\r") spanEnd -= 1;
          spans.push({ start: starts[li], end: spanEnd });
        }
      }
      return spans;
    }

    // Refuse ANY resulting .py content that does not parse. Round-6
    // lesson: gating only valid->invalid transitions let a file that was
    // BORN broken stay broken through repeated equally-broken "repairs".
    // The error names the line; if the file carries several errors they
    // must be fixed in one edit (or force=true saves partial progress).
    // ANY checker failure fails OPEN. Returns {ran, ok, detail} — `ran`
    // distinguishes a real verdict from a fail-open non-answer, so only
    // genuine passes feed the evidenced-good anchor.
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

    // Post-mutation structural check (P22 Tier 3) — a SENTENCE, never a
    // gate. This tool is where the lesson was learned: gemma's partial-
    // overlap edit at 13:40:08 duplicated put() and left a dangling `else`;
    // edit_file reported "ok", the broken file's exit-2 satisfied the first
    // test assertion, and the model spent 31 minutes editing help text on a
    // file that had not parsed since. The verdict — including whether the
    // file parsed BEFORE the edit — closes the loop at the same frequency
    // the mutations happen. Fail OPEN everywhere; shared with write_file.
    function structuralCheckKind(path, contentText) {
      var pp = path.split("\\").join("/").toLowerCase();
      var base = pp.split("/").pop();
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
      if (path.indexOf("'") !== -1) return null;
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
        if (r.code === 127 || err.indexOf("command not found") !== -1) return null;
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
      if (prevChk === "ok") history = " It parsed BEFORE this edit — this edit introduced the break.";
      else if (prevChk === "bad") history = " It did not parse before this edit either.";
      return " STRUCTURE: " + path + " does NOT parse (" + verdict.tool + "): " + verdict.detail + "." + history +
        " This is information, not a block — the edit was applied exactly as sent. Fix that line with another edit_file.";
    }
    // LITERAL-ESCAPE VERDICT (P23), shell-checked files only — shared with
    // write_file (full design comment there). A physical line that is a
    // COMMENT and carries two or more literal backslash-n sequences is a
    // flattened block hiding behind a '#': sh gives comments no escape
    // semantics, so the file parses cleanly and the hidden code silently
    // never runs. A SENTENCE, never a gate and never a repair — converting on
    // a heuristic would ACTIVATE code the author may not have meant to run.
    // Comment scoping keeps printf/awk lines with legitimate literal \n out.
    // Fails open.
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
    var filePath = input.file_path || input.filePath || input.path || input.file;

    // old/new variants accept ONLY string values, so a boolean flag like
    // replace=true can never be mistaken for the replacement text.
    var oldStr;
    var oldNames = ["old_string", "old_str", "old_text", "search", "find", "target"];
    for (var oi = 0; oi < oldNames.length; oi++) {
      if (typeof input[oldNames[oi]] === "string") { oldStr = input[oldNames[oi]]; break; }
    }
    var newStr;
    var newNames = ["new_string", "new_str", "new_text", "replacement", "replace_with", "replace"];
    for (var ni = 0; ni < newNames.length; ni++) {
      if (typeof input[newNames[ni]] === "string") { newStr = input[newNames[ni]]; break; }
    }

    if (!filePath && oldStr === undefined && newStr === undefined) {
      return fail("edit_file failed: you must pass file_path, old_string AND new_string. Nothing was changed. Call edit_file again like: edit_file(file_path=\"D:/path/to/file.py\", old_string=\"<exact text currently in the file>\", new_string=\"<the replacement text>\")");
    }
    if (!filePath) {
      return fail("edit_file failed: missing file_path. Nothing was changed. Call edit_file again with file_path (the file to edit) plus old_string AND new_string.");
    }
    if (oldStr === undefined) {
      return fail("edit_file failed: missing old_string. Nothing was changed. Call edit_file again with old_string set to the EXACT text currently in " + filePath + " (include 2-3 surrounding lines to make it unique) and new_string set to its replacement.");
    }
    if (newStr === undefined) {
      return fail("edit_file failed: missing new_string. Nothing was changed. Call edit_file again with the same old_string and new_string set to the replacement text (it may be empty to delete the snippet).");
    }
    if (oldStr === "") {
      return fail("edit_file failed: old_string is empty. Nothing was changed. edit_file replaces an existing snippet; to create a file or replace its entire content, use write_file with the complete text.");
    }

    // The user's own declared prohibitions come FIRST — before the file is
    // even read, and well before the syntax gate writes .__chk temp files
    // beside the target.
    var invBlock = invariantRefusal(filePath, "EDIT");
    if (invBlock !== "") return fail(invBlock);

    var content;
    try {
      content = Nanna.readFile(filePath);
    } catch (e) {
      var readErr = String(e);
      if (readErr.length > 120) readErr = readErr.substring(0, 120) + "...";
      return fail("edit_file failed: could not read " + filePath + " (" + readErr + "). Nothing was changed. Check the path, or use write_file to create a new file.");
    }

    // Identical old/new: if the file ALREADY contains the text, the desired
    // state holds — succeed as a no-op so the model moves on instead of
    // spiraling (observed live: it "confirms" content instead of diffing).
    // If the text is absent, its memory of the file is stale — say so.
    if (oldStr === newStr) {
      var lfSame = oldStr.split("\r\n").join("\n");
      var present = content.indexOf(oldStr) >= 0
        || content.indexOf(lfSame) >= 0
        || content.indexOf(lfSame.split("\n").join("\r\n")) >= 0
        || findLooseSpans(content, oldStr).length > 0;
      if (present) {
        return { content: "No change needed — " + filePath + " already contains exactly that text. Continue to the next step.", success: true };
      }
      var nearSame = closestSnippet(content, oldStr);
      return fail("edit_file failed: old_string and new_string are identical AND that text is not in " + filePath + " — the file's real content differs from your memory. The file is UNCHANGED." + (nearSame === "" ? "" : "\nClosest ACTUAL text in the file:\n" + nearSame + "\n") + "Call read_file, copy the real text as old_string, and set new_string to your fix.");
    }

    // Try the exact text first, then retry with line-ending normalization in
    // BOTH directions (LF old_string vs CRLF file, and CRLF old_string vs LF
    // file). Only the matched snippet is touched — the rest of the file keeps
    // its own line endings. The replacement is converted to the matched
    // flavor so the edit does not introduce mixed endings.
    // The file's own line-ending style, judged once. A one-line old_string
    // always matches exactly, so the CRLF conversion below (which runs only
    // when the exact match fails) never touched the replacement: adding
    // lines after a one-line anchor — the commonest edit — left LF lines in a
    // CRLF file and reported success. Every path now writes in the file's style.
    var fileIsCrlf = content.indexOf("\r\n") >= 0 && content.split("\r\n").join("").indexOf("\n") < 0;
    function inFileEol(text) {
      if (!fileIsCrlf) return text;
      return text.split("\r\n").join("\n").split("\n").join("\r\n");
    }
    var needle = oldStr;
    var replacement = inFileEol(newStr);
    if (content.indexOf(needle) < 0) {
      var oldLf = oldStr.split("\r\n").join("\n");
      var oldCrlf = oldLf.split("\n").join("\r\n");
      if (content.indexOf(oldCrlf) >= 0) {
        needle = oldCrlf;
        replacement = newStr.split("\r\n").join("\n").split("\n").join("\r\n");
      } else if (content.indexOf(oldLf) >= 0) {
        needle = oldLf;
        replacement = newStr.split("\r\n").join("\n");
      }
    }

    var updated;
    var replaced;

    if (content.indexOf(needle) >= 0) {
      var count = 0;
      var pos = content.indexOf(needle);
      while (pos >= 0) {
        count++;
        pos = content.indexOf(needle, pos + needle.length);
      }

      // Accept string "true" as well: small models often stringify booleans,
      // and rejecting it here would loop them on the ambiguity error below.
      function flagSet(v) { return v === true || v === "true"; }
      var replaceAll = flagSet(input.replace_all) || flagSet(input.replaceAll) || flagSet(input.all) || input.replace === true;
      var occurrence = input.occurrence;
      if (occurrence === undefined) occurrence = input.occurence;
      if (occurrence === undefined) occurrence = input.nth;

      if (occurrence !== undefined) {
        occurrence = Math.floor(occurrence);
        if (!(occurrence >= 1 && occurrence <= count)) {
          return fail("edit_file failed: occurrence=" + occurrence + " is out of range — old_string matches " + count + " time(s) in " + filePath + ". The file is UNCHANGED. Pass occurrence between 1 and " + count + ".");
        }
      }

      if (count > 1 && !replaceAll && occurrence === undefined) {
        return fail("edit_file failed: found " + count + " matches for old_string in " + filePath + ". The file is UNCHANGED. Either include more surrounding lines in old_string to make it unique, or pass replace_all=true, or pass occurrence=<1.." + count + "> to pick one match.");
      }

      if (occurrence !== undefined) {
        var at = content.indexOf(needle);
        for (var k = 1; k < occurrence; k++) {
          at = content.indexOf(needle, at + needle.length);
        }
        updated = content.substring(0, at) + replacement + content.substring(at + needle.length);
        replaced = 1;
      } else if (replaceAll) {
        updated = content.split(needle).join(replacement);
        replaced = count;
      } else {
        var first = content.indexOf(needle);
        updated = content.substring(0, first) + replacement + content.substring(first + needle.length);
        replaced = 1;
      }
    } else {
      // Exact and line-ending matches failed: whitespace-tolerant fallback.
      var spans = findLooseSpans(content, oldStr);
      if (spans.length === 1) {
        var spanText = content.substring(spans[0].start, spans[0].end);
        var looseReplacement = inFileEol(newStr);
        if (spanText.indexOf("\r\n") >= 0) {
          looseReplacement = newStr.split("\r\n").join("\n").split("\n").join("\r\n");
        }
        updated = content.substring(0, spans[0].start) + looseReplacement + content.substring(spans[0].end);
        replaced = 1;
      } else if (spans.length > 1) {
        return fail("edit_file failed: " + spans.length + " places in " + filePath + " match old_string once indentation differences are ignored. The file is UNCHANGED. Include 1-2 more surrounding lines in old_string to make it unique.");
      } else {
        // Before giving up: the model may have pasted read_file's own output
        // back. That is not a mistake it can see — read_file has no unnumbered
        // format — so the product has to undo its own formatting. Tried only
        // AFTER the ordinary and loose matches have failed, so a genuine
        // tab-separated file is never reinterpreted.
        var recovered = false;
        var unnumbered = stripLineNumberBlock(oldStr);
        if (unnumbered !== null) unnumbered = inFileEol(unnumbered);
        if (unnumbered !== null && unnumbered !== oldStr && content.indexOf(unnumbered) !== -1) {
          var occurrences = content.split(unnumbered).length - 1;
          if (occurrences === 1) {
            glog("edit_file: matched after stripping read_file line numbers from old_string: " + filePath);
            updated = content.split(unnumbered).join(inFileEol(newStr));
            replaced = 1;
            recovered = true;
          }
        }
        // An explicit flag, not `replaced === 0`: `replaced` is declared
        // without an initializer, so it is undefined on this path and the
        // comparison silently fell through — leaving `updated` unset for the
        // code below instead of returning the failure.
        if (!recovered) {
          var head = oldStr.split("\r\n").join("\n").split("\n").slice(0, 3).join("\n");
          if (head.length > 120) head = head.substring(0, 120) + "...";

          // NO file echo here, deliberately.
          //
          // The review asked for the current content to be inlined on a miss,
          // to save the model a read_file round-trip. Two standing invariants
          // in this tool's own tests say otherwise: a failure message must
          // stay small enough for a 32k-window model to absorb, and it must
          // not dump file content at all. Both are tested and both predate
          // this change, so the recommendation is declined rather than
          // silently overturned — dumping a file into an error is the same
          // "one result claims the whole window" harm the read-stub threshold
          // exists to prevent.
          //
          // The valuable half of that finding is upstream anyway: the reason
          // the model kept missing was that it was pasting read_file's own
          // line-numbered output back, which is now stripped and retried
          // before this point.
          var near = closestSnippet(content, oldStr);
          var echo = near === "" ? "" : "\nClosest ACTUAL text in the file:\n" + near;

          // Say only what was measured. The old text asserted a CAUSE — "the
          // file's real content differs from your memory" — that nothing had
          // checked, and never named the path it actually resolved.
          return failEscalating(
            "miss:" + filePath,
            "edit_file failed: your old_string does not appear in the " + content.length +
              " bytes currently at " + filePath + ". The file is UNCHANGED and intact. You searched for:\n" +
              head + echo +
              "\nCall read_file, copy a distinctive line from it verbatim into old_string, and keep the edit targeted — do not rewrite the file.",
            "edit_file failed AGAIN on " + filePath + ": old_string still does not appear in the " +
              content.length + " bytes on disk. Repeating the same search will not start matching. " +
              "Read the file with read_file and copy a distinctive line from it verbatim, or edit a " +
              "different part of the file."
          );
        }
      }
    }

    var pyChk = null;
    if (input.force !== true) {
      pyChk = pythonSyntaxCheck(filePath, updated);
      if (pyChk.ran && !pyChk.ok) {
        glog("edit_file guard: EDIT REFUSED (invalid .py) " + filePath + " (" + pyChk.detail + ")");
        return fail("REFUSED — after this edit " + filePath + " would NOT be valid Python (" + pyChk.detail + "). The file is UNCHANGED. Fix new_string so the whole file parses (if the file has several errors, fix them all in this one edit), then retry.");
      }
    }

    // SUFFIXED-COPY REFUSAL, shared with write_file (full design comment there).
    //
    // Same lesson as the ratchet floor immediately below, learned the same way:
    // a guard that lives in only one tool does not protect the file, it just
    // decides which tool gets used to damage it. write_file refused a write to
    // './minidb.sh' 17 times while './minidb' existed — and the model moved to
    // edit_file, which had no such check, and made 24 successful edits to the
    // fork. 2029 bytes of work went into a file the acceptance tests never read.
    //
    // Narrow, exactly as in write_file: it fires only when the target's STEM is
    // itself an existing file, i.e. the original carries no extension of its
    // own. Sibling formats are untouched — config.json beside config.yaml,
    // tool.ts beside tool.js — because their stems are not files.
    var forkBase = filePath.split("\\").join("/");
    var forkSlash = forkBase.lastIndexOf("/");
    var forkName = forkSlash >= 0 ? forkBase.substring(forkSlash + 1) : forkBase;
    var forkDot = forkName.lastIndexOf(".");
    // Conventional variants (`.env.local`, `Dockerfile.dev`) are siblings,
    // not forks — the same exemption as write_file's guard.
    var forkVariants = ["local", "dev", "development", "prod", "production", "staging", "test", "example", "sample", "template", "dist", "override", "ci", "debug", "release"];
    if (forkDot > 0 && (forkName.charAt(0) === "." ||
        forkVariants.indexOf(forkName.substring(forkDot + 1).toLowerCase()) !== -1)) {
      forkDot = -1;
    }
    if (forkDot > 0) {
      var forkStem = forkName.substring(0, forkDot);
      var forkStemPath = forkSlash >= 0
        ? forkBase.substring(0, forkSlash + 1) + forkStem
        : forkStem;
      var forkStat = null;
      try { forkStat = Nanna.stat(forkStemPath); } catch (eFork) { forkStat = null; }
      if (forkStat && forkStat.is_file) {
        // Escalates on repetition — see the note in write_file. Half the
        // wasted fork attempts in the live run came through edit_file, so a
        // guard that only hardens on one of the two paths leaves the loop
        // fully open on the other.
        return failEscalating(
          hiwaterKey(filePath),
          "EDIT REFUSED — '" + filePath + "' is '" + forkStem + "' with an extension " +
            "added, and '" + forkStem + "' already exists (" + (forkStat.size || 0) + " bytes). " +
            "Nothing was edited. Improving this copy cannot help: whatever reads '" + forkStem +
            "' goes on reading it, so this edit never reaches anyone, no matter " +
            "how good it is.\nMake the change in '" + forkStemPath + "' itself.",
          "STOP — '" + filePath + "' WILL NEVER BE ACCEPTED. You have tried it repeatedly. " +
            "The only file that counts is '" + forkStemPath + "'. Send this exact call instead:\n" +
            "    edit_file(file_path=\"" + forkStemPath + "\", old_string=<exact current text>, new_string=<replacement>)\n" +
            "Do not add an extension. Do not try '" + filePath + "' again."
        );
      }
    }

    // The ratchet floor applies to the RESULT, not to the tool that produced it.
    //
    // write_file refuses a write below 30% of the floor base and, in the
    // refusal, points the model at edit_file — which had no floor at all,
    // because "deletion is edit_file's job". So the guard did not fail, it
    // HERDED: measured 2026-07-28, write_file refused 13 times, the model moved
    // to edit_file, and seven edits took a working minidb from 9691 bytes to
    // 3941. The score went 17/42 to 3/42. A floor one tool can walk under is
    // not a floor.
    //
    // Deletion is still this tool's job — removing a section, a function, a
    // whole feature all stay possible, and each individual edit here can be
    // arbitrarily large. What is refused is the RESULT falling under the same
    // line write_file defends — 30% of the last evidenced-good size (or of the
    // high-water mark when no check ever passed) — whether that takes one edit
    // or twenty. Anything above the floor is untouched, so ordinary editing
    // never sees this.
    var floorInfo = hiwaterFloorInfo(filePath);
    if (floorInfo.base > 500 && updated.length < content.length && updated.length < floorInfo.base * 0.3) {
      var floorStory = floorInfo.anchor === "good"
        ? "its last version that passed a structural check held " + floorInfo.base + " bytes"
        : "it has held " + floorInfo.base + " bytes";
      glog("edit_file guard: EDIT REFUSED (shrink floor) " + filePath + " result=" + updated.length + " current=" + content.length + " floorBase=" + floorInfo.base + " anchor=" + floorInfo.anchor);
      return {
        content: "EDIT REFUSED — the file was NOT modified and is fully intact. " +
          "This edit would leave " + filePath + " at " + updated.length + " bytes, but " +
          floorStory + " (" + Math.round(updated.length / floorInfo.base * 100) +
          "% of that). Deleting that much in one step is almost always an accident — a " +
          "stale old_string matching more than you meant, or a rewrite sent as an edit.\n" +
          "Nothing was lost: read_file " + filePath + " to see what is actually there, then " +
          "make the change you intended against the real current text. If you genuinely mean " +
          "to remove most of the file, do it in separate edits that each take out one named " +
          "section — that way a mistake costs one section instead of the whole file.",
        success: false
      };
    }

    try {
      Nanna.writeFile(filePath, updated);
    } catch (e2) {
      // Classify BEFORE truncating (P23): the cause is the tail, and the
      // advice that follows it must match the cause — a permission denial is
      // not something a retry can fix.
      return fail("edit_file failed writing " + filePath + " " +
        writeFailureNote(filePath, String(e2),
          "Retry the same edit_file call; if it fails again, read the file to verify its current state before editing."));
    }

    // Structural verdict on the result (P22). The .py gate above already
    // ran the real check — adopt its answer instead of spawning the
    // interpreter again; under force the generic check runs on the file.
    var prevChk = null;
    try {
      var peek = hiwaterEntryFor(hiwaterLoad(), filePath);
      if (peek && (peek.chk === "ok" || peek.chk === "bad")) prevChk = peek.chk;
    } catch (ePeek) {
      // No history — the sentence just has no before/after clause.
    }
    var checkKind = structuralCheckKind(filePath, updated);
    var verdict = null;
    if (checkKind === "py" && pyChk && pyChk.ran) {
      verdict = { ok: pyChk.ok, tool: "python ast", detail: pyChk.detail };
    } else if (checkKind) {
      verdict = runStructuralCheck(checkKind, filePath, updated);
    }

    // Anti-erosion ratchet sync, shared with write_file (full design
    // comment there). edit_file is a TRUSTED in-band mutator: recording
    // {hi, last} after each successful edit keeps write_file's high-water
    // guard armed across surgical edits — otherwise every edit looks
    // out-of-band and hands the next rewrite a fresh current-size floor (the
    // 2-call nibble+rewrite erosion loop from the verify round). Best-effort,
    // fails open. A passing verdict also rebases the structural anchor
    // write_file's shrink hold measures against (P23) — computed only on the
    // rebase, so ordinary failing/absent verdicts pay nothing.
    var editGoodSyms = null;
    if (verdict && verdict.ok) {
      try {
        editGoodSyms = symbolNames(updated);
      } catch (eSyms) {
        // No anchor is safer than a stale one — hiwaterRecord drops it.
        editGoodSyms = null;
      }
    }
    hiwaterRecord(filePath, updated.length, content.length, verdict, editGoodSyms);

    // A surgical edit that quietly deletes a top-level definition is the shape
    // that destroys work, and until now only the whole-file path said anything
    // about it — edit_file computed the definition set purely for write_file's
    // guard to measure against, and never looked at it itself.
    //
    // Informational, not a hold: it costs no extra round-trip, it works on
    // every file class the regex can see, and a refusal here has a bounce cost
    // the evidence does not yet justify (models answer write-side holds by
    // escalating to MORE rewriting, not less).
    var removalNote = "";
    try {
      var beforeSyms = symbolNames(content);
      var afterSyms = symbolNames(updated);
      if (beforeSyms.length > 0) {
        var goneNames = [];
        for (var bi = 0; bi < beforeSyms.length; bi++) {
          if (afterSyms.indexOf(beforeSyms[bi]) === -1) goneNames.push(beforeSyms[bi]);
        }
        if (goneNames.length > 0) {
          removalNote = " NOTE: this edit removed " + goneNames.length + " top-level definition(s): [" +
            goneNames.slice(0, 8).join(", ") + (goneNames.length > 8 ? ", …" : "") +
            "]. The write SUCCEEDED and is on disk. If that was deliberate, nothing to do; if not, " +
            "restore them with another targeted edit rather than rewriting the file.";
          glog("edit_file removal-note for " + filePath + ": removed=[" + goneNames.join(",") + "] kept=" + afterSyms.length);
        }
      }
    } catch (eRemoval) {
      // The regex cannot see this file class — no note, no failure.
      removalNote = "";
    }

    var structNote = structSentence(filePath, verdict, prevChk) +
      (input.force === true ? "" : escapedCommentNote(checkKind, updated));
    if (verdict && !verdict.ok) {
      glog("edit_file structure: " + filePath + " does NOT parse after edit (" + verdict.tool + "): " + verdict.detail + (prevChk === "ok" ? " [parsed before this edit]" : ""));
    } else if (verdict && verdict.ok && prevChk === "bad") {
      glog("edit_file structure: " + filePath + " parses again after edit (" + verdict.tool + ")");
    }

    // `success` stays true: on the write family it means "the bytes landed",
    // and three separate mechanisms downstream read it that way (the world
    // epoch bump, failure counting, and error routing). What was missing is
    // the OUTCOME — this tool has already run the file's real parser and knows
    // the edit broke it, and every consumer that only saw the flag recorded a
    // break as landed work.
    //
    // Only ever set when a checker actually ran and returned a verdict. An
    // absent, unrun or fail-open verdict leaves this off entirely, because a
    // false "broken" would suppress completion and drain the item's budget —
    // `sh -n` is documented to cry wolf on valid bash where /bin/sh is dash.
    // A before/after view of what the edit changed, so a user returning to hours of
    // unattended work can SEE each edit instead of "replaced 1 occurrence(s)". It is
    // observability for the timeline, never an approval step.
    //
    // Never split the whole file: in Boa, split() costs lines x length. The common
    // prefix and suffix are found with native 4 KiB substring compares, both ends are
    // snapped to line boundaries, and only the changed region (capped) is split.
    function editDiff(before, after, maxLines, maxChars) {
      var n = Math.min(before.length, after.length);
      var step = 4096;
      var p = 0;
      while (p + step <= n && before.substring(p, p + step) === after.substring(p, p + step)) p += step;
      while (p < n && before.charCodeAt(p) === after.charCodeAt(p)) p++;
      var limit = n - p;
      var s = 0;
      while (s + step <= limit && before.substring(before.length - s - step, before.length - s) === after.substring(after.length - s - step, after.length - s)) s += step;
      while (s < limit && before.charCodeAt(before.length - 1 - s) === after.charCodeAt(after.length - 1 - s)) s++;
      if (p === before.length && p === after.length) return null;
      var lineStart = p === 0 ? 0 : before.lastIndexOf("\n", p - 1) + 1;
      var endA = before.indexOf("\n", before.length - s);
      if (endA === -1) endA = before.length;
      var endB = after.indexOf("\n", after.length - s);
      if (endB === -1) endB = after.length;
      var lineNo = 1;
      var at = before.indexOf("\n");
      while (at !== -1 && at < lineStart) { lineNo++; at = before.indexOf("\n", at + 1); }
      function lines(text, from, to) {
        var region = text.substring(from, Math.min(to, from + maxChars));
        var out = region.length === 0 ? [] : region.split("\n");
        if (out.length > 1 && out[out.length - 1] === "") out.pop();
        for (var i = 0; i < out.length; i++) out[i] = out[i].replace(/\r$/, "");
        return { list: out.slice(0, maxLines), cut: to - from > maxChars || out.length > maxLines };
      }
      var removed = lines(before, lineStart, endA);
      var added = lines(after, lineStart, endB);
      return {
        start_line: lineNo,
        removed: removed.list,
        added: added.list,
        truncated: removed.cut || added.cut
      };
    }
    var result = { content: "Edited " + filePath + ": replaced " + replaced + " occurrence(s). File is now " + updated.length + " characters." + structNote + removalNote, success: true };
    var data = {};
    if (verdict) {
      data.structure = { parses: verdict.ok === true, tool: verdict.tool, detail: verdict.detail };
    }
    // 40 lines / 4000 chars per side: the daemon journals this view beside the
    // call's output, which it caps at 4000 bytes, and clamps a diff to the same
    // bound — matching it here keeps that clamp a guard, not the normal path.
    var diff = editDiff(content, updated, 40, 4000);
    if (diff) data.diff = diff;
    if (data.structure || data.diff) result.data = data;
    return result;
  }
}

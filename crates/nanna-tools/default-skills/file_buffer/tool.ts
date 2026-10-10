export default {
  name: "file_buffer",
  version: "0.1.8",
  output: "memory",
  description: "Write a LARGE file across MULTIPLE tool calls: append chunks of text one call at a time, then commit once to write the real file. Use this instead of write_file when a file is too long to write in one call. Sequence: file_buffer(action=\"append\", file_path, content) repeatedly in order from the top of the file, then file_buffer(action=\"commit\", file_path) to write it. action=\"show\" previews the pending buffer, action=\"clear\" discards it. The real file only changes on commit. Commit carries write_file's safety net: a shrinking commit over a file that changed since you last read it returns the file's current content to merge, a shrinking commit that deletes more top-level sections than it keeps is held ONCE with the removed names (commit the same buffer again to confirm), the previous version is parked at <file>.__prev__ and the richest earlier version at <file>.__best__, and the cheapest structural check runs on the result with its verdict appended.",
  parameters: {
    type: "object",
    properties: {
      action: { type: "string", enum: ["append", "commit", "show", "clear"], description: "REQUIRED. append = add the next chunk to the pending buffer; commit = write the whole buffer to file_path and clear it; show = preview the pending buffer; clear = discard the pending buffer." },
      file_path: { type: "string", description: "REQUIRED. The REAL file being built. The pending buffer is kept beside it until commit." },
      content: { type: "string", description: "REQUIRED for append: the NEXT chunk of the file, continuing exactly where the buffer ended. A newline is inserted between chunks automatically if missing." },
    },
    required: ["action", "file_path"]
  },
  execute: function(input) {
    // Structured failures, never throws: thrown script errors reach the
    // model under five stacked "Execution failed:" prefixes.
    function fail(message) {
      return { content: message, success: false };
    }

    // Guard events and commits are logged at INFO with byte counts so this
    // write path can be audited from the daemon log (P22 evidence: the
    // file_buffer commit that produced a 0/42 artifact left NO ratchet or
    // size line — a second full-overwrite path that no guard observably
    // covers is how a protection quietly stops applying). Best-effort.
    function glog(msg) {
      try { Nanna.log("info", msg); } catch (e) { /* logging is optional */ }
    }

    // Anti-erosion ratchet state, shared with write_file v0.1.15 (full
    // design comment lives there). Commit is a TRUSTED in-band mutator: its
    // shrink guard judges against the same floor write_file defends, and a
    // successful commit records {hi, last} so write_file's guard stays
    // armed instead of treating the change as out-of-band. Without this,
    // the park->repair->commit cycle was the open erosion route during .py
    // fault storms (verify-round blocker). All state I/O is best-effort and
    // fails OPEN.
    var HIWATER_STATE = ".nanna/write_hiwater.json";
    var HIWATER_MAX_ENTRIES = 200;
    function hiwaterNormKey(path) {
      var k = path.split("\\").join("/").toLowerCase();
      while (k.indexOf("./") === 0) k = k.substring(2);
      while (k.indexOf("//") !== -1) k = k.split("//").join("/");
      return k;
    }
    // Canonical key (P22): absolute spellings under the workspace root
    // collapse to the relative form — one file, one entry. Legacy
    // spellings are folded forward by hiwaterEntryFor (design comment in
    // write_file).
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
    // The coverage high-water park — a recovery copy like .__prev__,
    // rewritten wholesale, never judged by cross-call history.
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
    // rest follows the fresher entry); the alias is dropped so the
    // split-brain heals on the next save through the map.
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
    // shrink hold, below) — the same field and semantics write_file uses: a
    // follow-up commit whose removal set signs the same proceeds, and any
    // successful commit rebuilds the entry, clearing the signature so the
    // hold re-arms for the next removal event. Best-effort.
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
    function hiwaterRecord(path, newSize, prevSize, verdict, goodSyms) {
      try {
        var key = hiwaterKey(path);
        if (hiwaterExempt(key)) return;
        var map = hiwaterLoad();
        var entry = hiwaterEntryFor(map, path);
        var hi = newSize > prevSize ? newSize : prevSize;
        // Monotone while the file exists: the previous mark survives
        // regardless of who touched the file in between (out-of-band
        // changes fold in as evidence — full design comment in write_file).
        // prevSize < 0 signals a FORCE commit: a deliberate re-shape that
        // re-arms from the committed size, exactly write_file's force reset
        // — no fold, no carried evidence.
        if (prevSize >= 0 && entry && typeof entry.hi === "number" && isFinite(entry.hi) && entry.hi > hi) {
          hi = entry.hi;
        }
        // Rebuilt from scratch, which is also how the structural shrink
        // hold's `heldGut` signature clears itself: any successful commit
        // drops it and the hold re-arms for the next removal event.
        var next = { hi: hi, last: newSize, at: Date.now() };
        if (prevSize >= 0 && entry) {
          if (hiwaterGood(entry) > 0) {
            next.good = entry.good;
            next.goodAt = entry.goodAt || 0;
            if (Array.isArray(entry.goodSyms)) {
              next.goodSyms = entry.goodSyms;
            }
          }
          if (entry.chk === "ok" || entry.chk === "bad") next.chk = entry.chk;
          // Coverage high-water record travels with the ratchet: a force
          // commit resets it exactly as it resets `good`.
          if (typeof entry.bestSyms === "number" && isFinite(entry.bestSyms)) {
            next.bestSyms = entry.bestSyms;
            next.bestAt = entry.bestAt || 0;
          }
        }
        if (verdict) {
          next.chk = verdict.ok ? "ok" : "bad";
          if (verdict.ok) {
            next.good = newSize;
            next.goodAt = Date.now();
            // The anchor's definition set rebases WITH its size (P23): a set
            // left over from an older good version would name definitions
            // this commit legitimately removed. Always replaced, never
            // carried past a rebase.
            if (Array.isArray(goodSyms)) {
              next.goodSyms = goodSyms;
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

    // Read-recency marks, shared with write_file/read_file (P22: "you
    // cannot shrink what you have not seen" — design comment in
    // write_file). Fails OPEN throughout.
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
    // "seen" / "stale" / "never" — same three verdicts as write_file (full
    // design comment there): the hold below is identical for "stale" and
    // "never", the sentence it prints is not. Unknown → "seen" (fail open).
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

    // USER-DECLARED FILE INVARIANTS (P23), shared contract with write_file /
    // edit_file (full design comment in write_file). Durable prohibitions the
    // USER stated in chat are registered at plan time and consulted before
    // any mutation — for this tool that means the append that starts building
    // a file under a forbidden path as well as the commit that lands it. The
    // refusal quotes the user's own sentence back; a missing, unreadable or
    // malformed registry means NO invariants (fail open, silently); force
    // does NOT bypass, because only the user lifts a constraint.
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
    // `path` is always the REAL target file, never the .__buffer__ sidecar:
    // the constraint is about the artifact the user named, and the sidecar's
    // existence is tooling detail.
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
          // happen (exec), not on a write path.
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
          glog("file_buffer guard: " + verb + " REFUSED (declared invariant " + kind + " on '" + inv.glob + "') " + path);
          return verb + " REFUSED — " + path + " is under a path you declared off-limits" +
            (kind === "no_create_under" ? " for NEW files" : "") +
            (quoted === "" ? " (" + kind + " on `" + inv.glob + "`)" : ": \"" + quoted + "\"") +
            ". Nothing was written and the file on disk is unchanged. That is YOUR instruction (declared for " +
            scope + "), not a tool limitation, and it stays in force until you lift it in chat. " +
            "The fix belongs in the artifact you are producing — if something that READS " + path +
            " is failing, change the code it exercises, not " + path + ". If you believe this constraint " +
            "genuinely blocks the goal, ask_user about it instead of working around it.";
        }
        return "";
      } catch (e) {
        return "";
      }
    }

    // WRITE-FAILURE HONESTY (P23), shared with write_file / edit_file (full
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

    // Python syntax gate — same contract as write_file/edit_file: refuse
    // ANY invalid .py content (a file that never parses is never useful;
    // the error names the line so the model can fix it). Fails OPEN if the
    // checker is unavailable. Returns {ran, ok, detail} — `ran`
    // distinguishes a real verdict from a fail-open non-answer.
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
        var cmd = "python '" + chk + "' '" + newTmp + "'; rc=$?; rm -f '" + chk + "' '" + newTmp + "'; exit $rc";
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
    // gate; same machinery as write_file (design comment there). Fail OPEN.
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
      if (kind === "sh") { cmd = "sh -n '" + path + "'"; toolName = "sh -n"; }
      else if (kind === "bash") { cmd = "bash -n '" + path + "'"; toolName = "bash -n"; }
      else if (kind === "node") { cmd = "node --check '" + path + "'"; toolName = "node --check"; }
      else if (kind === "py") { cmd = "python -c 'import ast,sys; ast.parse(open(sys.argv[1], encoding=\"utf-8\").read())' '" + path + "'"; toolName = "python ast"; }
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
      if (prevChk === "ok") history = " It parsed BEFORE this commit — this commit introduced the break.";
      else if (prevChk === "bad") history = " It did not parse before this commit either.";
      return " STRUCTURE: " + path + " does NOT parse (" + verdict.tool + "): " + verdict.detail + "." + history +
        " This is information, not a block — the file holds exactly the committed buffer. Fix that line with edit_file.";
    }

    // Rewrite-delta announcement on commit, same bidirectional rule as
    // write_file (design comment there): removed sections, changed bodies
    // of pre-existing sections (cheap per-symbol content hash), and a
    // commit that more than doubled the file. Informational only.
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
    // Body-end rule shared with write_file: trim trailing top-level code
    // out of the span so appends after the last symbol never read as a
    // body change (design comment in write_file).
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
    // body map when the caller has one (the structural hold's pre-commit pass
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

    function lineCount(text) {
      var n = 1;
      for (var i = 0; i < text.length; i++) {
        if (text.charAt(i) === "\n") n++;
      }
      return n;
    }

    function lastLines(text, howMany) {
      var lines = text.split("\n");
      while (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
      var start = lines.length > howMany ? lines.length - howMany : 0;
      var tail = lines.slice(start).join("\n");
      if (tail.length > 300) tail = "..." + tail.substring(tail.length - 300);
      return tail;
    }

    var filePath = input.file_path || input.filePath || input.path || input.file;
    var action = input.action || input.mode || input.op;
    var chunk;
    var chunkNames = ["content", "text", "data", "chunk"];
    for (var ci = 0; ci < chunkNames.length; ci++) {
      if (typeof input[chunkNames[ci]] === "string") { chunk = input[chunkNames[ci]]; break; }
    }

    if (!filePath) {
      return fail("file_buffer failed: missing file_path. Nothing was changed. Call it again with file_path (the real file being built) and action (append/commit/show/clear).");
    }
    // A content chunk with no action is an append — the likeliest intent.
    if (!action && chunk !== undefined) action = "append";
    if (!action) {
      return fail("file_buffer failed: missing action. Nothing was changed. Sequence: file_buffer(action=\"append\", file_path, content) repeatedly in order, then file_buffer(action=\"commit\", file_path=\"" + filePath + "\").");
    }
    action = String(action).toLowerCase();

    // The user's own declared prohibitions come FIRST, on every action that
    // builds toward a mutation of the real file — before the syntax gate
    // writes .__chk temp files beside the target and before the buffer
    // sidecar is created. show/clear read or drop tooling state only.
    var isCommitAction = action === "commit" || action === "flush" || action === "save";
    var isAppendAction = action === "append" || action === "add" || action === "write";
    if (isCommitAction || isAppendAction) {
      var invBlock = invariantRefusal(filePath, isCommitAction ? "COMMIT" : "APPEND");
      if (invBlock !== "") return fail(invBlock);
    }

    var bufPath = filePath + ".__buffer__";
    var buffered = null;
    try {
      buffered = Nanna.readFile(bufPath);
    } catch (eNone) {
      // No pending buffer yet.
    }

    if (action === "append" || action === "add" || action === "write") {
      if (chunk === undefined) {
        return fail("file_buffer failed: append needs content. Nothing was changed. Call file_buffer(action=\"append\", file_path=\"" + filePath + "\", content=\"<the next chunk of the file>\").");
      }
      var joined;
      if (buffered === null || buffered === "") {
        joined = chunk;
      } else if (buffered.charAt(buffered.length - 1) === "\n" || chunk.charAt(0) === "\n") {
        joined = buffered + chunk;
      } else {
        joined = buffered + "\n" + chunk;
      }
      try {
        Nanna.writeFile(bufPath, joined);
      } catch (eW) {
        // Classify BEFORE truncating (P23): the cause is the tail, and the
        // advice that follows it must match the cause.
        return fail("file_buffer failed writing the buffer for " + filePath + " " +
          writeFailureNote(bufPath, String(eW), "Retry the same append."));
      }
      return {
        content: "Buffered: " + joined.length + " chars / " + lineCount(joined) + " lines pending for " + filePath + ". The buffer now ends with:\n" + lastLines(joined, 2) + "\nContinue with the NEXT lines via file_buffer(action=\"append\", ...), or finish with file_buffer(action=\"commit\", file_path=\"" + filePath + "\").",
        success: true
      };
    }

    if (action === "commit" || action === "flush" || action === "save") {
      if (buffered === null || buffered === "") {
        return fail("file_buffer failed: nothing is buffered for " + filePath + ". Nothing was changed. Append the file content first: file_buffer(action=\"append\", file_path, content), then commit.");
      }
      var existing;
      var existingLen = 0;
      var fileExists = false;
      var hwKeyC = hiwaterKey(filePath);
      var pyGate = { ran: false, ok: false, detail: "" };
      // The pre-commit definition pass. Computed once (when the structural
      // hold below needs it) and reused by the rewrite-note after the commit,
      // so each side is parsed exactly once per call.
      var preOldBodies = null;
      var preNewBodies = null;
      if (input.force !== true) {
        pyGate = pythonSyntaxCheck(filePath, buffered);
        if (pyGate.ran && !pyGate.ok) {
          var syntaxDetail = pyGate.detail;
          // Quote the offending line: a ready-made old_string for the
          // repair edit — regeneration is never the answer here.
          var lineQuote = "";
          if (syntaxDetail.indexOf("line ") === 0) {
            var colonAt = syntaxDetail.indexOf(":");
            if (colonAt > 5) {
              var lineNo = parseInt(syntaxDetail.substring(5, colonAt), 10);
              if (lineNo >= 1) {
                var bufLines = buffered.split("\n");
                if (lineNo <= bufLines.length) {
                  var lq = bufLines[lineNo - 1];
                  if (lq.length > 80) lq = lq.substring(0, 80);
                  lineQuote = " Line " + lineNo + " of the buffer is: `" + lq + "` — use exactly that as old_string.";
                }
              }
            }
          }
          glog("file_buffer guard: COMMIT REFUSED (invalid .py) " + filePath + " buffer=" + buffered.length + " chars (" + syntaxDetail + ")");
          return fail("COMMIT REFUSED — the buffered content for " + filePath + " is not valid Python (" + syntaxDetail + "). The real file is UNCHANGED and the buffer is KEPT." + lineQuote + " Your NEXT call must be edit_file(file_path=\"" + bufPath + "\", old_string=<the broken line>, new_string=<the fixed line>), then commit again. Do NOT regenerate the file.");
        }
        try {
          existing = Nanna.readFile(filePath);
          if (existing !== undefined && existing !== null) {
            fileExists = true;
            existingLen = existing.length;
          }
        } catch (eR) {
          // New file.
        }

        // P22 Tier 3: you cannot shrink what you have not seen — the same
        // guard as write_file (design comment there). The buffer is KEPT;
        // the reply carries the file's current content and counts as the
        // read, so the next commit proceeds.
        // Echo bound, same rule as write_file (design comment there) and
        // shared by BOTH holds below: under 64 KiB the full content ships and
        // counts as the read; over it, a loudly-truncated head ships, the
        // mark is NOT recorded, and the next commit bounces into an explicit
        // ranged read_file rather than a merge against a partial view.
        var ECHO_MAX = 65536;
        // Evaluated only once the cheap conditions hold, so an ordinary
        // commit still costs no stat and no state read.
        var seenVerdict = (fileExists && !hiwaterExempt(hwKeyC) && buffered.length < existingLen &&
            typeof existing === "string") ? readSeenVerdict(filePath) : "seen";
        if (seenVerdict !== "seen") {
          // The reason clause, and ONLY the reason clause, differs between
          // the two verdicts — the way forward below is the same either way.
          var staleWhy = seenVerdict === "never"
            ? "but you have NEVER read this file in this session — your buffer was built without ever seeing what the file holds, "
            : "but the file has CHANGED since you last read it — your buffer was built from a stale copy, ";
          if (existingLen <= ECHO_MAX) {
            readmarkPut(filePath);
            glog("file_buffer guard: stale-shrink echo for " + filePath + " (buffer " + buffered.length + " over " + existingLen + " bytes; read-mark verdict: " + seenVerdict + ")");
            return fail(
              "COMMIT HELD — the real file is UNCHANGED and the buffer is KEPT. Committing would shrink " + filePath +
              " from " + existingLen + " to " + buffered.length + " bytes, " + staleWhy +
              "so parts of the current file would be silently destroyed. " +
              "Here is the CURRENT content of " + filePath + ":\n\n" + existing +
              "\n\nCompare it with your buffer (file_buffer action=\"show\"), fold anything missing into the buffer with " +
              "edit_file(file_path=\"" + bufPath + "\", ...), then commit again. This reply counts as your read — the commit will not be held for this reason again."
            );
          }
          glog("file_buffer guard: stale-shrink hold (truncated echo) for " + filePath + " (buffer " + buffered.length + " over " + existingLen + " bytes; read-mark verdict: " + seenVerdict + ")");
          return fail(
            "COMMIT HELD — the real file is UNCHANGED and the buffer is KEPT. Committing would shrink " + filePath +
            " from " + existingLen + " to " + buffered.length + " bytes, " + staleWhy +
            "so parts of the current file would be silently destroyed. " +
            "The file is too large (" + existingLen + " bytes) to echo here; its first lines are:\n\n" + existing.substring(0, 4096) +
            "\n\n[TRUNCATED — only the first 4096 of " + existingLen + " bytes shown; the file on disk is complete and unaffected.] " +
            "This truncated preview does NOT count as reading the file. Call read_file(\"" + filePath + "\") " +
            "(with offset/limit for ranges) to see the current content, fold anything missing into the buffer with " +
            "edit_file(file_path=\"" + bufPath + "\", ...), then commit again — after a real read the commit will be accepted."
          );
        }

        // Shrink floor, same anchor as write_file: 30% of the last
        // evidenced-good size, falling back to the monotone high-water mark
        // when no structural check has ever passed. A commit that does not
        // shrink the current file is never refused.
        var hwEntry = hiwaterEntryFor(hiwaterLoad(), filePath);
        var hwBase = existingLen;
        var hwHi = hiwaterHi(hwEntry);
        if (hwHi > hwBase) hwBase = hwHi;
        var hwGoodC = hiwaterGood(hwEntry);
        var floorBase = hwGoodC > 0 ? hwGoodC : hwBase;
        if (!hiwaterExempt(hwKeyC) &&
            floorBase > 500 && buffered.length < existingLen && buffered.length < floorBase * 0.3) {
          var sizeStory;
          if (hwGoodC > 0) {
            sizeStory = "holds " + existingLen + " now, and its last version that passed a structural check held " + floorBase;
          } else if (hwBase > existingLen) {
            sizeStory = "holds " + existingLen + " now and has held " + hwBase + " before";
          } else {
            sizeStory = "currently holds " + existingLen;
          }
          glog("file_buffer guard: COMMIT REFUSED (shrink floor) " + filePath + " buffer=" + buffered.length + " existing=" + existingLen + " floorBase=" + floorBase + " anchor=" + (hwGoodC > 0 ? "good" : "hi"));
          return fail("COMMIT REFUSED — the buffer holds only " + buffered.length + " chars but " + filePath + " " + sizeStory + ". The file is UNCHANGED and the buffer is KEPT — it looks incomplete. Keep appending the rest of the file, then commit again.");
        }

        // STRUCTURAL SHRINK HOLD (P23), same guard write_file carries (full
        // design comment there): bytes cannot see function removal, so a
        // SHRINKING commit that removes more pre-existing definitions than it
        // keeps — or drops definitions present in the last version that
        // PASSED a structural check — is held ONCE, with the removed names
        // and the current content as merge material, and the removal set
        // signed onto the ratchet entry. Commit the same buffer again and it
        // lands; any successful commit clears the signature. A guard that
        // lives in only one tool does not protect the file, it just decides
        // which tool gets used to damage it (the 2026-07-28 lesson). Fails
        // OPEN in every direction; the buffer is always KEPT.
        if (fileExists && !hiwaterExempt(hwKeyC) && buffered.length < existingLen &&
            typeof existing === "string") {
          var gutHold = null;
          try {
            preOldBodies = symbolBodies(existing);
            preNewBodies = symbolBodies(buffered);
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
              // Measured against the buffered content, not against disk — the
              // current disk copy may itself already have lost something the
              // evidenced-good version had.
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
                // One removal SET, deduplicated: the two arms overlap
                // whenever a definition is missing from both disk and the
                // good anchor.
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
            // No structural opinion — the commit proceeds exactly as before.
            gutHold = null;
          }
          if (gutHold) {
            heldGutPut(filePath, gutHold.sig);
            var gutWhy = "This commit would replace " + filePath + " (" + existingLen + " → " + buffered.length +
              " bytes) and REMOVE " + gutHold.names.length + " of the " + gutHold.old +
              " top-level sections the file defines, keeping " + gutHold.kept + ": " + nameList(gutHold.names) + "." +
              (gutHold.fromGood.length > 0
                ? " " + nameList(gutHold.fromGood) + " were present in the last version of this file that PASSED a structural check."
                : "") +
              " Deleting more than you keep is usually a buffer built from a stale or partial copy rather than a deliberate removal, and byte counts cannot see it.";
            var gutHow = "If those sections are genuinely obsolete, commit this SAME buffer again and it will be written — " +
              "this hold fires once per removal set, not once per attempt. Otherwise fold the missing sections back in with " +
              "edit_file(file_path=\"" + bufPath + "\", ...) and commit again.";
            glog("file_buffer guard: structural shrink hold for " + filePath + " (" + existingLen + "->" + buffered.length +
              " bytes) removed=[" + gutHold.names.join(",") + "] kept=" + gutHold.kept + " sig=" + gutHold.sig);
            if (existingLen <= ECHO_MAX) {
              readmarkPut(filePath);
              return fail(
                "COMMIT HELD — the real file is UNCHANGED and the buffer is KEPT. " + gutWhy +
                "\n\nHere is the CURRENT content of " + filePath + ":\n\n" + existing +
                "\n\n" + gutHow + " This reply counts as your read."
              );
            }
            return fail(
              "COMMIT HELD — the real file is UNCHANGED and the buffer is KEPT. " + gutWhy +
              "\n\nThe file is too large (" + existingLen + " bytes) to echo here; its first lines are:\n\n" +
              existing.substring(0, 4096) +
              "\n\n[TRUNCATED — only the first 4096 of " + existingLen + " bytes shown; the file on disk is complete and unaffected.] " +
              "This truncated preview does NOT count as reading the file — call read_file(\"" + filePath +
              "\") (with offset/limit for ranges) to see what those sections contain. " + gutHow
            );
          }
        }

        // P22: displaced content stays recoverable — park the outgoing
        // version at <file>.__prev__ before the overwrite, exactly as
        // write_file does. Best-effort, never blocks the commit.
        if (fileExists && typeof existing === "string" && existing !== buffered && !hiwaterExempt(hwKeyC)) {
          try { Nanna.writeFile(filePath + ".__prev__", existing); } catch (ePrev) { /* best-effort */ }
        }
      }
      try {
        Nanna.writeFile(filePath, buffered);
      } catch (eC) {
        // Classify BEFORE truncating (P23): the cause is the tail, and the
        // advice that follows it must match the cause — a permission denial
        // is not something a retry can fix.
        return fail("file_buffer failed writing " + filePath + " — the buffer is KEPT. " +
          writeFailureNote(filePath, String(eC), "Retry the commit."));
      }
      try {
        Nanna.exec("rm -f '" + bufPath + "' '" + filePath + ".__cleared__'", null, 15);
      } catch (eRm) {
        try { Nanna.writeFile(bufPath, ""); } catch (eZ) { /* leftovers are harmless */ }
      }

      // Structural verdict on what landed (P22): the .py gate above already
      // ran the real check — adopt its answer; otherwise the generic check
      // runs on the committed file. The verdict also feeds the ratchet's
      // evidenced-good anchor.
      var prevChk = null;
      var verdict = null;
      if (!hiwaterExempt(hwKeyC)) {
        try {
          var peek = hiwaterEntryFor(hiwaterLoad(), filePath);
          if (peek && (peek.chk === "ok" || peek.chk === "bad")) prevChk = peek.chk;
        } catch (ePeek) {
          // No history — the sentence just has no before/after clause.
        }
        var checkKind = structuralCheckKind(filePath, buffered);
        if (checkKind === "py" && pyGate.ran) {
          verdict = { ok: pyGate.ok, tool: "python ast", detail: pyGate.detail };
        } else if (checkKind) {
          verdict = runStructuralCheck(checkKind, filePath, buffered);
        }
      }
      if (input.force === true || !fileExists) prevChk = null;

      // In-band ratchet sync, same semantics as write_file. On a force
      // commit `existing` was never read: prevSize -1 re-arms from the
      // committed size and drops the stale good/chk evidence — exactly
      // force's reset semantics in write_file.
      // A passing verdict also rebases the structural anchor the shrink hold
      // measures against (P23) — computed only on the rebase, and reusing the
      // hold's pass over this exact content when it already ran. No anchor is
      // safer than a stale one, so a failed pass records none.
      var commitGoodSyms = null;
      if (verdict && verdict.ok) {
        try {
          if (!preNewBodies) preNewBodies = symbolBodies(buffered);
          commitGoodSyms = symbolNames(buffered, preNewBodies);
        } catch (eSyms) {
          commitGoodSyms = null;
        }
      }
      hiwaterRecord(filePath, buffered.length, input.force === true ? -1 : existingLen, verdict, commitGoodSyms);
      // The file now holds exactly the committed buffer — the session's
      // knowledge of the content is current, so the commit counts as a
      // read for the stale-shrink guard.
      if (!hiwaterExempt(hwKeyC)) readmarkPut(filePath);

      var structNote = structSentence(filePath, verdict, prevChk);
      if (verdict && !verdict.ok) {
        glog("file_buffer structure: " + filePath + " does NOT parse after commit (" + verdict.tool + "): " + verdict.detail);
      }

      // Rewrite-delta note, bidirectional (design comment in write_file).
      var lossNote = "";
      var bestPath = filePath + ".__best__";
      if (fileExists && input.force !== true && typeof existing === "string" && existing !== buffered) {
        try {
          var oldBodies = preOldBodies || symbolBodies(existing);
          var newBodies = preNewBodies || symbolBodies(buffered);
          var dropped = [];
          var changed = [];
          for (var sName in oldBodies) {
            if (newBodies[sName] === undefined) dropped.push(sName);
            else if (newBodies[sName] !== oldBodies[sName]) changed.push(sName);
          }
          // COVERAGE HIGH-WATER PARK (P23), same rule as write_file (design
          // comment there): .__prev__ holds one generation, so a two-commit
          // spiral rotates the good version out. The outgoing version is
          // parked at .__best__ whenever its top-level definition COUNT beats
          // the recorded record (not a subset relation — spirals rename
          // symbols); only commits that actually REMOVE sections qualify, the
          // first parks unconditionally, and a force commit resets the record
          // with the ratchet. Best-effort; never blocks.
          var bestNote = "";
          if (dropped.length > 0 && !hiwaterExempt(hwKeyC)) {
            var outCount = 0;
            for (var oName in oldBodies) outCount++;
            try {
              var bMap = hiwaterLoad();
              var bEntry = hiwaterEntryFor(bMap, filePath);
              var bHave = bEntry && typeof bEntry.bestSyms === "number" && isFinite(bEntry.bestSyms)
                ? bEntry.bestSyms : 0;
              if (outCount > bHave) {
                Nanna.writeFile(bestPath, existing);
                if (!bEntry) bEntry = { hi: existingLen, last: buffered.length, at: Date.now() };
                bEntry.bestSyms = outCount;
                bEntry.bestAt = Date.now();
                bMap[hwKeyC] = bEntry;
                hiwaterSave(bMap);
                glog("file_buffer parked coverage high-water for " + filePath + " at " + bestPath +
                  " (" + outCount + " sections, previous record " + bHave + ")");
              } else if (bHave > 0) {
                bestNote = " The fullest prior version (" + bHave + " sections, against " + outCount +
                  " in the version this commit replaced) is parked at " + bestPath +
                  " — read_file it before rewriting again.";
              }
            } catch (eBest) {
              // Recovery copies are best-effort.
            }
          }
          var grewPastDouble = existingLen > 500 && buffered.length > existingLen * 2;
          if (dropped.length > 0 || changed.length > 0 || grewPastDouble) {
            var parts = "";
            if (dropped.length > 0) parts += " It REMOVED sections that existed on disk before: " + nameList(dropped) + ".";
            if (changed.length > 0) parts += " It CHANGED the bodies of pre-existing sections: " + nameList(changed) + ".";
            if (grewPastDouble) {
              parts += dropped.length === 0 && changed.length === 0
                ? " It more than doubled the file while keeping every pre-existing section's body intact — the growth is purely additive."
                : " It also more than doubled the file — most of what is on disk now is new text.";
            }
            lossNote = " NOTE: this commit replaced " + filePath + " (" + existingLen + " → " + buffered.length + " bytes)." + parts +
              " The commit SUCCEEDED — the file holds exactly the committed buffer. If any of those sections were verified working, re-verify them now; the previous version is preserved at " + filePath + ".__prev__ (read_file it to recover anything)." + bestNote;
            glog("file_buffer rewrite-note for " + filePath + " (" + existingLen + "->" + buffered.length + " bytes): removed=[" + dropped.join(",") + "] changed=[" + changed.join(",") + "]" + (grewPastDouble ? " grew>2x" : ""));
          }
        } catch (eLoss) {
          // Informational only.
        }
      }

      glog("file_buffer commit: " + filePath + " " + (fileExists ? existingLen : 0) + "->" + buffered.length + " bytes (" + lineCount(buffered) + " lines)" + (input.force === true ? " [force]" : ""));
      return {
        content: "Committed " + buffered.length + " chars (" + lineCount(buffered) + " lines) to " + filePath + ". Buffer cleared." + structNote + lossNote + " Verify the file now with exec, then continue.",
        success: true
      };
    }

    if (action === "show" || action === "preview" || action === "status") {
      if (buffered === null || buffered === "") {
        return { content: "Buffer for " + filePath + " is empty. Start with file_buffer(action=\"append\", file_path, content).", success: true };
      }
      return {
        content: "Pending buffer for " + filePath + ": " + buffered.length + " chars / " + lineCount(buffered) + " lines. Ends with:\n" + lastLines(buffered, 10),
        success: true
      };
    }

    if (action === "clear" || action === "discard" || action === "reset") {
      // Friction on serial discards (round-11 lesson: the model deleted
      // parked drafts to regenerate — first via shell rm, and clear would
      // be the same loop with a different name). The first discard per
      // file is free; the second requires force, and the refusal steers
      // back to the one-line repair.
      var clearMarker = filePath + ".__cleared__";
      // Nothing pending: succeed as a no-op instead of steering the model
      // at a draft that does not exist (ship-gate finding).
      if (buffered === null || buffered === "") {
        return { content: "Buffer for " + filePath + " is already empty — nothing to discard. Continue.", success: true };
      }
      if (input.force !== true) {
        var clearedBefore = null;
        try {
          clearedBefore = Nanna.readFile(clearMarker);
        } catch (eMk) {
          // First discard.
        }
        if (clearedBefore !== null && clearedBefore !== undefined) {
          var steer = "";
          if (buffered !== null && buffered !== "") {
            var clearChk = pythonSyntaxCheck(filePath, buffered);
            if (clearChk.ran && !clearChk.ok) {
              steer = " The current draft has exactly one blocking error (" + clearChk.detail + ") — fixing that one line is faster than regenerating.";
            }
          }
          return fail("CLEAR REFUSED — you already discarded one draft for " + filePath + "; discarding again is the regeneration loop." + steer + " Repair the draft: edit_file(file_path=\"" + bufPath + "\", old_string=<the broken line>, new_string=<the fix>), then file_buffer(action=\"commit\").");
        }
        try { Nanna.writeFile(clearMarker, "1"); } catch (eWk) { /* best effort */ }
      }
      try {
        Nanna.exec("rm -f '" + bufPath + "'", null, 15);
      } catch (eRm2) {
        try { Nanna.writeFile(bufPath, ""); } catch (eZ2) { /* best effort */ }
      }
      return { content: "Buffer for " + filePath + " discarded. The real file was not touched. NOTE: this file cannot be discarded again — repair the next draft instead of regenerating it.", success: true };
    }

    return fail("file_buffer failed: unknown action '" + action + "'. Use append (add the next chunk), commit (write the file), show (preview), or clear (discard). Nothing was changed.");
  }
}

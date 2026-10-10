export default {
  name: "exec",
  version: "0.1.7",
  output: "memory",
  // Script-engine deadline (seconds). This is only a backstop for a hung script:
  // the shell bridge owns the real per-command timeout — the `timeout` parameter
  // below, or an auto-detected 30s/120s — and kills the child when it fires. This
  // ceiling must stay ABOVE the bridge's widest auto-detect (120s) so a long
  // build/VCS command is never preempted by the engine (which would orphan the
  // child). A larger explicit `timeout` extends this deadline automatically.
  timeout: 180,
  description: "Execute a shell command in a POSIX bash shell (Git Bash on Windows, sh on Unix) and return its output. ALWAYS bash syntax: pipes, &&, ||, [ -f x ] / [ -d x ], ls, cat/grep/tail, mkdir -p, 2>/dev/null, forward-slash paths. NEVER cmd.exe syntax — 'if exist', '2>nul', 'cd /d', 'errorlevel' all FAIL here. To search code, use the code_search tool — rg/ripgrep is not guaranteed on PATH. Use for build commands, scripts, git operations, etc. After a command that redirects into a script or JSON file, the cheapest structural check runs on that file and its verdict is appended to the output.",
  parameters: {
    type: "object",
    properties: {
      command: { type: "string", description: "Shell command to execute. Runs in the workspace directory by default." },
      workdir: { type: "string", description: "Working directory for the command. Defaults to the workspace directory if omitted." },
      timeout: { type: "integer", description: "Timeout in seconds. Default: 30 for simple commands, 120 for git/cargo/npm/build tools (auto-detected). Override with a specific value if needed." }
    },
    required: ["command"]
  },
  execute: function(input) {
    // Accept multiple parameter name variants from different models
    if (!input.command) {
      input.command = input.cmd || input.script || input.shell || input.bash_command || input.shell_command;
    }
    if (!input.command) {
      // Every other file-touching skill names itself, says nothing ran, and
      // lists what it accepts. This one did none of those, so a model that
      // passed the wrong key learned nothing about which keys work.
      return {
        content: "exec: no command given, so nothing ran. Pass the command as `command` (also accepted: cmd, script, shell, bash_command, shell_command), e.g. exec({command: \"ls -la\"}).",
        success: false
      };
    }

    // Guard events are logged at INFO so they can be audited from the
    // daemon log after a run (P22). Best-effort, never blocks.
    function glog(msg) {
      try { Nanna.log("info", msg); } catch (e) { /* logging is optional */ }
    }

    // --- anti-erosion: shell redirection over a ratchet-protected file ---
    //
    // Mirrors write_file/edit_file's key normalization exactly, or the
    // lookup misses and the guard silently does nothing. (The ratchet keys
    // on "./minidb" and "minidb" as the same file; so must this.)
    var HIWATER_STATE = ".nanna/write_hiwater.json";
    function hiwaterNormKey(path) {
      var k = path.split("\\").join("/").toLowerCase();
      while (k.indexOf("./") === 0) k = k.substring(2);
      while (k.indexOf("//") !== -1) k = k.split("//").join("/");
      return k;
    }
    // Canonical key (P22): absolute spellings under the workspace root
    // collapse to the relative form — one file, one entry (the observed
    // split-brain let an absolute-path mutation dodge the guard). The
    // legacy spelling is still consulted at lookup so old entries protect.
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
    function hiwaterMap() {
      try {
        var raw = Nanna.readFile(HIWATER_STATE);
        var parsed = JSON.parse(raw);
        if (parsed && typeof parsed === "object") return parsed;
      } catch (e) {
        // No state, or unreadable: nothing is protected yet. Fail OPEN.
      }
      return {};
    }
    // The entry under the canonical key, folded with the legacy spelling
    // (hi = max of both) so neither spelling escapes the guard.
    function hiwaterEntryFor(map, path) {
      var canon = hiwaterKey(path);
      var alias = hiwaterNormKey(path);
      var a = map[canon] || null;
      var b = alias !== canon ? (map[alias] || null) : null;
      if (!a) return b;
      if (b) {
        var hiA = typeof a.hi === "number" && isFinite(a.hi) ? a.hi : 0;
        var hiB = typeof b.hi === "number" && isFinite(b.hi) ? b.hi : 0;
        if (hiB > hiA) a.hi = hiB;
      }
      return a;
    }
    function hiwaterEntryHi(entry) {
      return entry && typeof entry.hi === "number" && isFinite(entry.hi) ? entry.hi : 0;
    }

    // Every redirection target in the command, with whether it appends.
    // `>>` appends (safe from clobbering, still a mutation); `>` and `>|`
    // truncate; `tee` truncates without -a/--append.
    function redirectTargets(command) {
      var targets = [];
      var i = 0;
      while (i < command.length) {
        var ch = command.charAt(i);
        if (ch === ">") {
          var append = false;
          var j = i + 1;
          if (command.charAt(j) === ">") { append = true; j++; }
          if (!append && command.charAt(j) === "|") j++;   // `>|` force-clobber
          while (j < command.length && (command.charAt(j) === " " || command.charAt(j) === "\t")) j++;
          var tok = "";
          while (j < command.length) {
            var c = command.charAt(j);
            if (c === " " || c === "\t" || c === "\n" || c === ";" || c === "&" ||
                c === "|" || c === "<" || c === ">" || c === ")") break;
            tok += c;
            j++;
          }
          // Strip surrounding quotes the model may have added.
          if (tok.length >= 2) {
            var f = tok.charAt(0), l = tok.charAt(tok.length - 1);
            if ((f === '"' && l === '"') || (f === "'" && l === "'")) tok = tok.substring(1, tok.length - 1);
          }
          if (tok.length > 0) targets.push({ path: tok, append: append });
          i = j;
          continue;
        }
        i++;
      }
      // `tee FILE` truncates just like `>`; `tee -a FILE` appends. Matched
      // as a standalone token, every occurrence: the old bare indexOf
      // false-fired on any word ENDING in "tee " (committee, guarantee,
      // employee), pushing the NEXT word as a phantom target — which could
      // refuse an innocent `echo "committee minidb" > log.txt` outright and,
      // worse, feed the post-command structural check a file the command
      // never touched, silently rewriting its evidenced-good anchor
      // (ultrareview on PR #224). A real tee sits in command position after
      // a pipe or separator, so the boundary set is exactly those.
      var teeScan = 0;
      while (true) {
        var teeAt = command.indexOf("tee ", teeScan);
        if (teeAt === -1) break;
        teeScan = teeAt + 4;
        if (teeAt > 0) {
          var teeBefore = command.charAt(teeAt - 1);
          if (teeBefore !== " " && teeBefore !== "\t" && teeBefore !== "\n" &&
              teeBefore !== "|" && teeBefore !== ";" && teeBefore !== "&" && teeBefore !== "(") continue;
        }
        var rest = command.substring(teeAt + 4);
        var teeAppend = false;
        if (rest.indexOf("-a ") === 0) { teeAppend = true; rest = rest.substring(3); }
        else if (rest.indexOf("--append ") === 0) { teeAppend = true; rest = rest.substring(9); }
        while (rest.charAt(0) === " ") rest = rest.substring(1);
        var tok2 = rest.split(" ")[0].split("\n")[0].split("|")[0].split(";")[0].trim();
        if (tok2.length > 0 && tok2.charAt(0) !== "-") targets.push({ path: tok2, append: teeAppend });
      }
      return targets;
    }

    // Targets of CLOBBERING redirects only (append cannot clobber).
    function clobberTargets(command) {
      var all = redirectTargets(command);
      var out = [];
      for (var i = 0; i < all.length; i++) {
        if (!all[i].append) out.push(all[i].path);
      }
      return out;
    }

    // Deletion/rename of a ratchet-protected file is the OTHER laundering
    // verb: the high-water is monotone only while the file exists, and the
    // model discovered `rm -f ./minidb` 48 seconds after a refused shrink —
    // erasing a 7965-byte high-water so the next fragment write passed
    // (observed 2026-08-09; repeats the redirect lesson through rm).
    function destructiveTargets(command) {
      var targets = [];
      // Split on shell separators, walk word by word; after an `rm`/`unlink`
      // token, every non-flag word is a target. After `mv`, the FIRST
      // non-flag word is (the source moves away — same erasure).
      var words = command.split(/[;&|\n]+/).join(" \n ").split(/[ \t]+/);
      var mode = null;
      var mvTaken = false;
      // The verb only counts in COMMAND position (first word of a
      // statement): `grep rm notes.txt` names rm as data, not as intent.
      var cmdPos = true;
      for (var w = 0; w < words.length; w++) {
        var word = words[w];
        if (word === "") continue;
        if (word === "\n") { mode = null; mvTaken = false; cmdPos = true; continue; }
        if (cmdPos) {
          cmdPos = false;
          if (word === "rm" || word === "unlink") { mode = "rm"; continue; }
          if (word === "mv") { mode = "mv"; mvTaken = false; continue; }
          mode = null;
          continue;
        }
        if (mode === null) continue;
        if (word.charAt(0) === "-") continue;
        var tok = word;
        if (tok.length >= 2) {
          var f0 = tok.charAt(0), l0 = tok.charAt(tok.length - 1);
          if ((f0 === '"' && l0 === '"') || (f0 === "'" && l0 === "'")) tok = tok.substring(1, tok.length - 1);
        }
        if (mode === "rm") { targets.push(tok); }
        else if (mode === "mv" && !mvTaken) { targets.push(tok); mvTaken = true; }
      }
      return targets;
    }

    // Nanna's own records under .nanna/ — the ratchet ledger, the read
    // marks, the declared-invariant registry. write_file refuses rewriting the
    // ledger ("would silently disarm the erosion guard"); exec let
    // `rm -f .nanna/write_hiwater.json` or `echo {} > …` through, resetting
    // every file's high-water, and `rm .nanna/declared_invariants.json`
    // dropped every rule the user had declared.
    function isNannaState(raw) {
      var k = hiwaterKey(raw);
      while (k.length > 1 && k.charAt(k.length - 1) === "/") k = k.substring(0, k.length - 1);
      return k === ".nanna" || k.indexOf(".nanna/") === 0 || k.indexOf("/.nanna/") !== -1 ||
        (k.length > 7 && k.lastIndexOf("/.nanna") === k.length - 7);
    }

    // The user's declared invariants (write_file's registry, same format and
    // matching): `no_delete` and `read_only` are enforced HERE, where deletion
    // and shell rewrites happen — write_file says so and leaves them to exec,
    // and before this exec never read the registry at all. Missing or
    // malformed registry => no rules (fail open), as in write_file.
    var INVARIANT_STATE = ".nanna/declared_invariants.json";
    function invariantsLoad() {
      try {
        var raw = Nanna.readFile(INVARIANT_STATE);
        if (!raw) return [];
        var parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== "object" || !Array.isArray(parsed.invariants)) return [];
        return parsed.invariants;
      } catch (e) {
        return [];
      }
    }
    // Mirrors write_file's invariantMatches exactly, or one rule would match
    // differently depending on which tool touched the file.
    function invariantMatches(glob, canonPath, normPath) {
      try {
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

    function stateOrInvariantRefusal(command) {
      var clobbers = clobberTargets(command);
      var deletes = destructiveTargets(command);
      var all = clobbers.concat(deletes);
      for (var a = 0; a < all.length; a++) {
        if (all[a].indexOf("$") !== -1) continue;
        if (isNannaState(all[a])) {
          glog("exec guard: NOT EXECUTED (Nanna state " + all[a] + "): " + (command.length > 160 ? command.substring(0, 160) + "..." : command));
          return "NOT EXECUTED — nothing was changed. " + all[a] + " is Nanna's own record (the " +
            "write ledger, read marks, or the rules you declared), not project output; " +
            "overwriting, moving or deleting it would silently switch off the safety checks " +
            "and the user's declared rules. Leave .nanna/ alone and continue with your task.";
        }
      }
      var list = invariantsLoad();
      if (list.length === 0) return null;
      for (var t = 0; t < all.length; t++) {
        var raw = all[t];
        if (raw.indexOf("$") !== -1 || raw.indexOf("/dev/") === 0) continue;
        var deleting = t >= clobbers.length;
        var canon = hiwaterKey(raw);
        var norm = hiwaterNormKey(raw);
        for (var i = 0; i < list.length; i++) {
          var inv = list[i];
          if (!inv || typeof inv !== "object" || typeof inv.glob !== "string") continue;
          var kind = typeof inv.kind === "string" ? inv.kind : "";
          var applies = kind === "read_only" || (kind === "no_delete" && deleting);
          if (kind === "no_create_under" && !deleting) {
            var exists = false;
            try { exists = !!Nanna.stat(raw); } catch (eS) { exists = false; }
            applies = !exists;
          }
          if (!applies || !invariantMatches(inv.glob, canon, norm)) continue;
          var quoted = typeof inv.source === "string" && inv.source !== "" ? inv.source : "";
          glog("exec guard: NOT EXECUTED (declared invariant " + kind + " on '" + inv.glob + "') " + raw);
          return "NOT EXECUTED — nothing was changed. This command would " +
            (deleting ? "delete or move " : "write ") + raw + ", which is under a path you declared " +
            "off-limits" + (quoted === "" ? " (" + kind + " on `" + inv.glob + "`)" : ": \"" + quoted + "\"") +
            ". That is YOUR instruction, not a tool limitation, and it stays in force until you lift " +
            "it in chat. If you believe it genuinely blocks the goal, call lift_invariant with glob `" +
            inv.glob + "` — it asks the user — instead of working around it.";
        }
      }
      return null;
    }

    function redirectClobberRefusal(command) {
      var targets = clobberTargets(command);
      var destructive = destructiveTargets(command);
      for (var d = 0; d < destructive.length; d++) targets.push(destructive[d]);
      if (targets.length === 0) return null;
      var map = hiwaterMap();
      for (var t = 0; t < targets.length; t++) {
        var raw = targets[t];
        // A redirect into a variable or process substitution is not a path
        // we can reason about; leave it alone.
        if (raw.indexOf("$") !== -1 || raw.indexOf("/dev/") === 0) continue;
        var hi = hiwaterEntryHi(hiwaterEntryFor(map, raw));
        if (hi > 500) {
          glog("exec guard: NOT EXECUTED (clobber/delete of ratchet-protected " + raw + ", hi=" + hi + "): " + (command.length > 160 ? command.substring(0, 160) + "..." : command));
          return "NOT EXECUTED — the file was NOT modified and is fully intact. " +
            "This command would replace or delete " + raw + ", which is a file you built with " +
            "write_file/edit_file and which has held " + hi + " bytes. Redirecting over it, " +
            "rm-ing it, or mv-ing it away destroys the WHOLE file with no safety check, and " +
            "that is how a working script gets silently replaced by a shorter one that drops " +
            "features you already had passing.\n\n" +
            "Nothing failed and nothing is lost — " + raw + " is exactly as it was.\n\n" +
            "To change part of it: edit_file(file_path=\"" + raw + "\", old_string=<exact current text>, new_string=<replacement>).\n" +
            "To replace all of it on purpose: read_file(\"" + raw + "\") first, merge your change " +
            "into the FULL text, then write_file with the complete content.\n" +
            "Redirect to a DIFFERENT path if you meant to write scratch output.";
        }
      }
      return null;
    }

    var denylist = [
      "rm -rf /",
      "rm -rf /*",
      "format C:",
      "mkfs",
      "dd if=/dev/zero",
      ":(){ :|:& };:"
    ];

    var cmdLower = input.command.toLowerCase().trim();
    for (var i = 0; i < denylist.length; i++) {
      if (cmdLower.indexOf(denylist[i].toLowerCase()) === 0) {
        return "Error: Command blocked by safety check: \"" + denylist[i] + "\"";
      }
    }

    // This shell is Git Bash, not cmd.exe. Catch unambiguous cmd.exe syntax
    // BEFORE running it so the model gets a correction instead of a cryptic
    // bash parse error (observed live: small models mix shells). Markers are
    // chosen to avoid matching legitimate bash that merely mentions the
    // words (e.g. grep for "errorlevel" in code is fine; "if errorlevel "
    // is the cmd.exe conditional).
    var cmdisms = ["if exist ", "if not exist ", "cd /d ", "if errorlevel "];
    var cmdism = null;
    for (var j = 0; j < cmdisms.length; j++) {
      if (cmdLower.indexOf(cmdisms[j]) !== -1) { cmdism = cmdisms[j].trim(); break; }
    }
    if (!cmdism) {
      // ">nul" only counts as cmd.exe when it stands alone — not as a
      // prefix of a real filename like ">nul_check.txt".
      var nulAt = cmdLower.indexOf(">nul");
      while (nulAt !== -1) {
        var afterNul = cmdLower.charAt(nulAt + 4);
        if (afterNul === "" || afterNul === " " || afterNul === "\t" || afterNul === "&" || afterNul === "2") {
          cmdism = ">nul";
          break;
        }
        nulAt = cmdLower.indexOf(">nul", nulAt + 4);
      }
    }
    // Buffer files are MANAGED state: deleting one with the shell defeats
    // the parked-draft repair loop (observed live: rm -f '*.__buffer__'
    // then regenerate, forever). The legal doors are edit_file + commit,
    // or file_buffer action="clear".
    if ((cmdLower.indexOf("rm ") !== -1 || cmdLower.indexOf("del ") !== -1 || cmdLower.indexOf("unlink") !== -1) && cmdLower.indexOf(".__buffer__") !== -1) {
      return {
        content: "NOT EXECUTED — buffer files (*.__buffer__) are managed drafts; do not delete them with the shell. Either REPAIR the draft (edit_file the broken line, then file_buffer action=\"commit\") or discard it properly with file_buffer(action=\"clear\", file_path=<the real file>).",
        success: false
      };
    }
    if (!cmdism && cmdLower.indexOf("type ") === 0 && (cmdLower.indexOf(":\\") !== -1 || cmdLower.indexOf(":/") !== -1)) {
      // cmd.exe `type <file>` prints a file; bash `type` describes a
      // command, so `type D:\...` just echoes the path back (observed
      // live twice — the model believed it had read the file).
      return {
        content: "NOT EXECUTED — in bash, 'type' does not print files (that is cmd.exe). Use: cat \"" + (input.command.split(" ").slice(1).join(" ").split("|")[0].trim()) + "\" instead. Then call exec again.",
        success: false
      };
    }
    if (cmdism) {
      return {
        content: "NOT EXECUTED — exec runs Git Bash (POSIX), not cmd.exe, and your command contains cmd.exe syntax ('" + cmdism + "'). Rewrite with bash: '[ -d path ]' / '[ -f path ]' to test existence, 'ls' to list, '2>/dev/null' to silence errors, 'mkdir -p' to create dirs. Then call exec again.",
        success: false
      };
    }

    // Shell redirection is an UNGUARDED WRITE PATH, and it silently defeats
    // the anti-erosion ratchet that write_file/edit_file enforce.
    //
    // Observed live 2026-07-28, qwen3.5:9b on the 42-feature ladder: the
    // model built ./minidb up to 8620 bytes through write_file/edit_file,
    // then ran `cat > ./minidb << 'EOF'` with a ~1.7 KB script. That write
    // took no 30% floor check AND left the ratchet's `last` stale, so the
    // next write_file saw disk != last, re-based to disk truth, and the
    // 8620-byte high-water collapsed to 1719. From there every subsequent
    // shrink was "acceptable" relative to the laundered peak, and the run
    // fell from 19/42 to 6/42 while looking healthy the whole way.
    //
    // Scoped deliberately: only files the ratchet is ALREADY protecting are
    // refused. A redirect to out.txt, a temp file, or anything the agent has
    // never built with write_file has no ratchet entry and passes straight
    // through — including the ladder's own `sh tests/test_NN.sh >out.txt`.
    // So this closes the erosion hole without touching ordinary shell work.
    var clobber = redirectClobberRefusal(input.command);
    if (clobber) {
      return { content: clobber, success: false };
    }
    var declared = stateOrInvariantRefusal(input.command);
    if (declared) {
      return { content: declared, success: false };
    }

    // Bridge failures (spawn errors, bad workdir, missing files) must be
    // RETURNED, not thrown — a thrown error reaches the model under five
    // stacked "Execution failed:" prefixes (observed live: os error 267 in
    // a retry loop). Name the likely cause so the retry is a correction.
    var result;
    try {
      result = Nanna.exec(input.command, input.workdir, input.timeout);
    } catch (e) {
      var bridgeErr = String(e && e.message ? e.message : e);
      // A deadline is not a spawn failure. The script engine's own ceiling
      // (this skill's `timeout: 180`) can only fire AFTER the command has been
      // running for that long, so "Nothing ran" is exactly backwards — and it
      // is the sentence the model remembers when it decides whether to re-run
      // a command that already did half its work. Matched on the full message,
      // before the truncation below, so a long error can never hide it.
      if (bridgeErr.indexOf("Timeout after") !== -1) {
        return {
          content: "exec hit its deadline (" + bridgeErr + ") and the command was killed. It " +
            "RAN for that whole time — this is NOT a failure to start — and it may have left " +
            "side effects; disk is truth, so check with ls/cat before re-running. Any output " +
            "it had produced was lost with the kill. Re-run something narrower, or pass a " +
            "larger `timeout`.",
          success: false
        };
      }
      if (bridgeErr.length > 200) bridgeErr = bridgeErr.substring(0, 200);
      var hint = "";
      if (bridgeErr.indexOf("directory name is invalid") !== -1 || bridgeErr.indexOf("os error 267") !== -1) {
        hint = " The workdir you passed ('" + (input.workdir || "") + "') is not a valid directory — pass an existing DIRECTORY (not a file path), or omit workdir to use the workspace default.";
      } else if (bridgeErr.indexOf("cannot find the file") !== -1 || bridgeErr.indexOf("os error 2") !== -1) {
        hint = " A path in the command or workdir does not exist — check it with ls, then retry.";
      }
      return {
        content: "exec could not start the command (" + bridgeErr + ")." + hint + " Nothing ran; retry with the correction.",
        success: false
      };
    }

    // A deadline-exceeded run comes back as a RESULT, not a throw: the bridge
    // kills the tree and hands over what the command had already produced.
    // Killing it does not un-run it, and the failure line the model remembers
    // is what decides whether the next turn re-runs a command that already
    // created half its files. A bridge that carries no `timed_out` field never
    // enters this branch, so the behaviour there is exactly as before.
    var timedOut = result.timed_out === true;

    var output = result.stdout;
    if (result.stderr) {
      output += output ? "\n" : "";
      output += "--- stderr ---\n" + result.stderr;
    }

    if (timedOut) {
      var elapsedText = typeof result.elapsed_ms === "number" && isFinite(result.elapsed_ms)
        ? "ran for " + (result.elapsed_ms / 1000).toFixed(1) + "s, killed at the "
        : "was killed at the ";
      // The deadline the caller ASKED for. When it was omitted the bridge
      // auto-detects one from the command, so quote the rule rather than
      // guessing a number this side cannot know.
      var askedTimeout = parseInt(input.timeout, 10);
      var deadlineText = isFinite(askedTimeout) && askedTimeout > 0
        ? askedTimeout + "s deadline"
        : "auto-detected deadline (30s, or 120s for git/cargo/npm/build tools)";
      output = "TIMED OUT — " + elapsedText + deadlineText + " — the command executed and may " +
        "have left side effects; disk is truth. Whatever it wrote, created or deleted before " +
        "the kill is still there: check with ls/cat before re-running, then narrow the command " +
        "or raise `timeout`." +
        (output ? "\n\nOutput captured before the kill:\n" + output
                : "\n\nIt had produced no output when the deadline fired.");
    } else if (!result.success) {
      output = "Command failed (exit code " + result.code + ")\n" + output;
    }

    // Post-mutation structural check (P22 Tier 3): a redirect is a write
    // path, so it gets the same honesty as write_file/edit_file — the
    // cheapest applicable check runs on each redirected-into file and the
    // verdict is appended AS A SENTENCE, never a gate. Appends included:
    // `echo ... >> ./minidb` bypasses the byte ratchet by construction (it
    // only grows the file) yet can render it unparseable, and did, unremarked
    // (2026-08-10 ministral workspace). Only the first 3 targets are checked
    // so the checks (15s cap each) can never crowd the 180s engine deadline
    // and orphan the child process. Fail OPEN on every path.
    function structuralCheckKindForDisk(path, contentText) {
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
    try {
      var rTargets = redirectTargets(input.command);
      var seenTargets = {};
      var checked = 0;
      for (var rt = 0; rt < rTargets.length && checked < 3; rt++) {
        var tPath = rTargets[rt].path;
        if (seenTargets[tPath]) continue;
        seenTargets[tPath] = true;
        if (tPath.indexOf("$") !== -1 || tPath.indexOf("/dev/") === 0) continue;
        var tKey = hiwaterKey(tPath);
        // Managed sidecars and the ledger itself are not checkable work files.
        if (tKey.lastIndexOf(".__buffer__") === tKey.length - 11 && tKey.length >= 11) continue;
        if (tKey.lastIndexOf(".__prev__") === tKey.length - 9 && tKey.length >= 9) continue;
        if (tKey.lastIndexOf("write_hiwater.json") !== -1) continue;
        var tStat = null;
        try { tStat = Nanna.stat(tPath); } catch (eSt) { tStat = null; }
        if (!tStat || !tStat.is_file) continue;
        var tContent = null;
        var kind = structuralCheckKindForDisk(tPath, null);
        if (kind === null) {
          // Extensionless: classify by shebang. The read is bounded — a
          // structural check exists for human-scale source files, and
          // anything bigger is data (read_file's own cap is 10MB; 512KB is
          // far past any source file these checks understand).
          var baseT = tPath.split("\\").join("/").split("/").pop();
          if (baseT.indexOf(".") === -1 && tStat.size <= 524288) {
            try { tContent = Nanna.readFile(tPath); } catch (eRd) { tContent = null; }
            if (tContent !== null) kind = structuralCheckKindForDisk(tPath, tContent);
          }
        } else if (kind === "json") {
          // Engine-side parse needs the bytes; same scope bound as above.
          if (tStat.size > 1048576) { kind = null; }
          else {
            try { tContent = Nanna.readFile(tPath); } catch (eRd2) { kind = null; }
          }
        }
        if (!kind) continue;
        var v = runStructuralCheck(kind, tPath, tContent);
        if (!v) continue;
        checked++;
        // Before/after clause from the ratchet's recorded verdict, when the
        // file has one; a pass also refreshes the evidenced-good anchor for
        // files the ratchet already tracks (never creates new entries —
        // exec is not an in-band author).
        var prevChk = null;
        try {
          var ledger = hiwaterMap();
          var lEntry = hiwaterEntryFor(ledger, tPath);
          if (lEntry && (lEntry.chk === "ok" || lEntry.chk === "bad")) prevChk = lEntry.chk;
          if (lEntry) {
            lEntry.chk = v.ok ? "ok" : "bad";
            if (v.ok) {
              lEntry.good = tStat.size;
              lEntry.goodAt = Date.now();
            }
            ledger[hiwaterKey(tPath)] = lEntry;
            var alias2 = hiwaterNormKey(tPath);
            if (alias2 !== hiwaterKey(tPath)) delete ledger[alias2];
            Nanna.writeFile(HIWATER_STATE, JSON.stringify(ledger));
          }
        } catch (eLg) {
          // Ledger sync is best-effort.
        }
        if (v.ok) {
          output += "\n\nSTRUCTURE: " + tPath + " parses (" + v.tool + ")" +
            (prevChk === "bad" ? " — the earlier syntax break is fixed." : ".");
        } else {
          var hist = "";
          if (prevChk === "ok") hist = " It parsed BEFORE this command — this command introduced the break.";
          else if (prevChk === "bad") hist = " It did not parse before this command either.";
          output += "\n\nSTRUCTURE: " + tPath + " does NOT parse after this command (" + v.tool + "): " + v.detail + "." + hist +
            " This is information, not a block — the command's output landed as shown above.";
          glog("exec structure: " + tPath + " does NOT parse after redirect (" + v.tool + "): " + v.detail);
        }
      }
    } catch (eStruct) {
      // The verdict is informational only — never affects the result.
    }

    // A killed command never succeeded, whatever the bridge recorded for a run
    // that has no exit status of its own.
    return { content: output || "(no output)", success: timedOut ? false : result.success };
  }
}

export default {
  name: "python",
  requires: ["python.exec"],
  version: "0.1.1",
  output: "memory",
  description: "Execute Python code or manage saved scripts. Embedded interpreter — no system Python required. Supports standard library (os, json, re, pathlib, collections, math, etc). No pip/third-party packages. Use for file manipulation, data processing, batch edits, text transforms, and scripting.",
  parameters: {
    type: "object",
    properties: {
      code: {
        type: "string",
        description: "Python code to execute. Multi-line supported. Use print() for output. Not needed if running a saved script."
      },
      action: {
        type: "string",
        description: "Script management action: 'run' (default), 'save', 'load', 'list', 'delete'. 'run' executes code directly. 'save' stores code in the session-scoped script registry, not as a workspace file — it creates no file of that name on disk; use write_file for that. 'load' runs a saved script. 'list' shows all saved scripts. 'delete' removes a saved script from that registry.",
        enum: ["run", "save", "load", "list", "delete"]
      },
      name: {
        type: "string",
        description: "Script name for save/load/delete actions. Use short descriptive names like 'fix_imports' or 'batch_rename'."
      },
      args: {
        type: "string",
        description: "JSON string of arguments to pass to the script. Available as `args` dict in the Python code. Example: '{\"pattern\": \"*.rs\", \"dry_run\": true}'"
      },
      workdir: {
        type: "string",
        description: "Working directory for the script. Defaults to the session workspace."
      },
      timeout: {
        type: "integer",
        description: "Timeout in seconds. Default: 30"
      }
    },
    required: []
  },
  execute: function(input) {
    var action = input.action || "run";
    var sessionId = Nanna.sessionId();
    var scriptFile = sessionId ? ".nanna-scripts-" + sessionId + ".json" : ".nanna-scripts.json";
    // Set by the 'load' case so the run below can say WHERE the code came
    // from. A saved script executes with no trace of its origin otherwise.
    var loadedFrom = null;

    // Load saved scripts
    var scripts = {};
    try {
      var content = Nanna.readFile(scriptFile);
      if (content) {
        scripts = JSON.parse(content);
      }
    } catch (e) {
      scripts = {};
    }

    switch (action) {
      case "save": {
        if (!input.name) {
          return { content: "Error: 'name' is required for save action", success: false };
        }
        if (!input.code) {
          return { content: "Error: 'code' is required for save action", success: false };
        }
        scripts[input.name] = {
          code: input.code,
          saved_at: new Date().toISOString(),
          description: input.description || ""
        };
        Nanna.writeFile(scriptFile, JSON.stringify(scripts, null, 2));
        var lineCount = input.code.split("\n").length;
        // A side-effect ack must name WHERE the effect landed. "Saved script
        // 'build.sh'" reads as a file write, and a model that believes it
        // wrote build.sh never writes it — the effect landed in a session
        // JSON index, under a key, not at that path. Say so in the ack, since
        // nothing downstream can see the difference.
        return {
          content: "Saved script '" + input.name + "' (" + lineCount + " lines) to the session " +
            "script registry (" + scriptFile + ") — this stores code for python(action='load', " +
            "name='" + input.name + "'); it did NOT create or modify any workspace file named '" +
            input.name + "'.",
          success: true
        };
      }

      case "list": {
        var names = Object.keys(scripts);
        if (names.length === 0) {
          return {
            content: "No saved scripts in the session script registry (" + scriptFile +
              "). This lists saved code, not workspace files — use list_dir for those.",
            success: true
          };
        }
        var lines = ["Saved scripts (" + names.length + ") in the session script registry (" +
          scriptFile + "):"];
        for (var i = 0; i < names.length; i++) {
          var s = scripts[names[i]];
          var lineCount = s.code.split("\n").length;
          var desc = s.description ? " — " + s.description : "";
          lines.push("  • " + names[i] + " (" + lineCount + " lines, saved " + s.saved_at + ")" + desc);
        }
        return { content: lines.join("\n"), success: true };
      }

      case "delete": {
        if (!input.name) {
          return { content: "Error: 'name' is required for delete action", success: false };
        }
        if (!scripts[input.name]) {
          return {
            content: "Error: script '" + input.name + "' not found in the session script registry (" +
              scriptFile + ")",
            success: false
          };
        }
        delete scripts[input.name];
        Nanna.writeFile(scriptFile, JSON.stringify(scripts, null, 2));
        return {
          content: "Deleted script '" + input.name + "' from the session script registry (" +
            scriptFile + ") — no workspace file was deleted.",
          success: true
        };
      }

      case "load": {
        if (!input.name) {
          return { content: "Error: 'name' is required for load action", success: false };
        }
        if (!scripts[input.name]) {
          var available = Object.keys(scripts);
          var hint = available.length > 0 ? " Available: " + available.join(", ") : " No scripts saved.";
          return {
            content: "Error: script '" + input.name + "' not found in the session script registry (" +
              scriptFile + ")." + hint,
            success: false
          };
        }
        // Fall through to execution with the saved code
        input.code = scripts[input.name].code;
        loadedFrom = input.name;
        // Fall through to run
      }

      case "run":
      default: {
        if (!input.code) {
          return { content: "Error: 'code' is required. Provide Python code to execute, or use action='load' with a script name.", success: false };
        }

        // Prepend args injection if provided
        var codeToRun = input.code;
        if (input.args) {
          codeToRun = "import json\nargs = json.loads(" + JSON.stringify(input.args) + ")\n" + codeToRun;
        }

        var result;
        try {
          result = Nanna.service("python.exec", {
            code: codeToRun,
            // The documented default — the session workspace. Unset, the
            // interpreter ran in the daemon's own working directory.
            workdir: input.workdir || Nanna.workdir() || undefined,
            timeout: input.timeout || 30
          });
        } catch (e) {
          return { content: "Error: Python service unavailable: " + e, success: false };
        }

        var output = "";
        if (result.stdout) {
          output = result.stdout;
        }
        if (result.stderr) {
          output += (output ? "\n--- stderr ---\n" : "") + result.stderr;
        }
        if (result.error) {
          output += (output ? "\n--- error ---\n" : "") + result.error;
        }
        if (!output) {
          output = "(no output)";
        }

        if (result.duration_ms !== undefined) {
          output += "\n(" + result.duration_ms + "ms)";
        }

        // Same rule as the save ack: name where this code came from, so a
        // loaded script's output is never mistaken for a workspace file's.
        if (loadedFrom !== null) {
          output = "(ran saved script '" + loadedFrom + "' from the session script registry " +
            scriptFile + " — not a workspace file)\n" + output;
        }

        return { content: output, success: result.success };
      }
    }
  }
}

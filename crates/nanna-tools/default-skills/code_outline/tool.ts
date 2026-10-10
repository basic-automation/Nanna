export default {
  name: "code_outline",
  version: "0.1.1",
  output: "memory",
  description: "Generate a structural outline of a source file showing functions, classes, structs, and other definitions. Useful for understanding file structure without reading the entire file.",
  parameters: {
    type: "object",
    properties: {
      path: { type: "string", description: "Path to the source file" }
    },
    required: ["path"]
  },
  execute: function(input) {
    var filePath = input.path || input.file_path || input.filePath || input.file;
    if (!filePath) throw "Missing required parameter: path";
    var content = Nanna.readFile(filePath);
    var lines = content.split("\n");
    var ext = filePath.split(".").pop().toLowerCase();

    var patterns = getPatterns(ext);
    var outline = [];

    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];
      for (var j = 0; j < patterns.length; j++) {
        if (patterns[j].test(line)) {
          outline.push((i + 1) + ": " + line.trimEnd());
          break;
        }
      }
    }

    var pct = lines.length > 0 ? Math.round((1 - outline.length / lines.length) * 100) : 0;
    var header = "# Outline of " + input.path + " (" + lines.length + " -> " + outline.length + " lines, " + pct + "% reduction)\n";

    return header + outline.join("\n");
  }
}

function getPatterns(ext) {
  switch (ext) {
    case "rs":
      return [
        // `pub(crate)` / `pub(super)` and the fn qualifiers (`const`, `async`,
        // `unsafe`, `extern "C"`): `pub\s+` alone skipped every restricted
        // item, so an outline of a daemon module listed none of its
        // `pub(crate) async fn`s.
        /^\s*(pub(\([^)]*\))?\s+)?((const|async|unsafe|default)\s+|extern\s+"[^"]*"\s+)*fn\s+/,
        /^\s*(pub(\([^)]*\))?\s+)?(unsafe\s+)?(struct|enum|trait|impl|type|const|static|mod|use|union)\b/,
        /^\s*#\[derive/,
        /^\s*#\[cfg/
      ];
    case "py":
      return [
        /^\s*(async\s+)?def\s+/,
        /^\s*class\s+/,
        /^import\s+/,
        /^from\s+.*import\s+/,
        /^\s*@\w+/
      ];
    case "ts":
    case "tsx":
    case "js":
    case "jsx":
      return [
        /^\s*(export\s+)?(default\s+)?(async\s+)?function\s+/,
        /^\s*(export\s+)?(default\s+)?class\s+/,
        /^\s*(export\s+)?(const|let|var)\s+\w+\s*[:=]\s*(async\s+)?\(/,
        /^\s*(export\s+)?interface\s+/,
        /^\s*(export\s+)?type\s+\w+\s*=/,
        /^\s*(export\s+)?enum\s+/,
        /^\s*import\s+/
      ];
    case "go":
      return [
        /^func\s+/,
        /^type\s+\w+\s+(struct|interface)/,
        /^package\s+/,
        /^import\s+/,
        /^var\s+/,
        /^const\s+/
      ];
    default:
      return [
        /^\s*(pub\s+)?(fn|function|def|class|struct|enum|trait|impl|type|interface|module|package)\s+/,
        /^\s*(import|from|use|require)\s+/
      ];
  }
}

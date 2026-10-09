export default {
  name: "analyze_image",
  requires: ["vision.analyze"],
  version: "0.1.0",
  output: "memory",
  description: "Analyze an image using a vision model. Describe contents, answer questions, or extract information from images.",
  parameters: {
    type: "object",
    properties: {
      path: { type: "string", description: "Path to the image file" },
      prompt: { type: "string", description: "What to analyze or ask about the image. Default: 'Describe this image in detail'" }
    },
    required: ["path"]
  },
  execute: function(input) {
    try {
      // Resolved and permission-checked by the bridge (a relative path is the
      // workspace's); the service itself takes the path as given.
      var file;
      try {
        file = Nanna.stat(input.path).path;
      } catch (e) {
        return "Error: cannot read " + input.path + ": " + e;
      }
      var result = Nanna.service("vision.analyze", {
        path: file,
        prompt: input.prompt || "Describe this image in detail"
      });
      return result.text || result.description || "(no analysis returned)";
    } catch (e) {
      return "Error: Vision service not available. " + e;
    }
  }
}

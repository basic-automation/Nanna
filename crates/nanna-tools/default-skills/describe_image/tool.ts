export default {
  name: "describe_image",
  requires: ["vision.analyze"],
  version: "0.1.0",
  output: "memory",
  description: "Get a concise description of an image. Useful for accessibility, captioning, or quick understanding of visual content.",
  parameters: {
    type: "object",
    properties: {
      path: { type: "string", description: "Path to the image file" }
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
        prompt: "Provide a brief, factual description of this image in 1-3 sentences."
      });
      return result.text || result.description || "(no description returned)";
    } catch (e) {
      return "Error: Vision service not available. " + e;
    }
  }
}

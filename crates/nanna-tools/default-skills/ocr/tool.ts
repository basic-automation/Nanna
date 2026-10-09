export default {
  name: "ocr",
  requires: ["vision.analyze"],
  version: "0.1.0",
  output: "memory",
  description: "Extract text from an image using optical character recognition. Works on screenshots, photos of documents, handwriting, etc.",
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
        prompt: "Extract ALL text visible in this image. Reproduce the text exactly as it appears, preserving layout and formatting where possible. If no text is found, say 'No text detected'."
      });
      return result.text || "(no text detected)";
    } catch (e) {
      return "Error: Vision/OCR service not available. " + e;
    }
  }
}

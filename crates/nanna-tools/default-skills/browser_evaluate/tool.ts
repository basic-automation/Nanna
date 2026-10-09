export default {
  name: "browser_evaluate",
  requires: ["browser.evaluate"],
  version: "0.1.0",
  output: "memory",
  description: "Evaluate JavaScript code in a browser page context. Useful for extracting dynamic data or manipulating page state.",
  parameters: {
    type: "object",
    properties: {
      url: { type: "string", description: "URL of the page to evaluate in" },
      expression: { type: "string", description: "JavaScript expression to evaluate in the page" }
    },
    required: ["url", "expression"]
  },
  execute: function(input) {
    try {
      var result = Nanna.service("browser.evaluate", {
        url: input.url,
        expression: input.expression
      });
      return "Result: " + (result.value !== undefined ? JSON.stringify(result.value) : "(undefined)");
    } catch (e) {
      return { content: "Error: " + e, success: false };
    }
  }
}

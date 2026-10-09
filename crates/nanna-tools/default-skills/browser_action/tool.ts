export default {
  name: "browser_action",
  requires: ["browser.action"],
  version: "0.1.0",
  output: "memory",
  description: "Perform an action in a browser page: click, type, scroll, or navigate. Useful for web automation and testing.",
  parameters: {
    type: "object",
    properties: {
      url: { type: "string", description: "URL of the page (or current page if session exists)" },
      action: {
        type: "string",
        output: "context",
  description: "Action: 'click', 'type', 'scroll', 'navigate', 'wait'",
        enum: ["click", "type", "scroll", "navigate", "wait"]
      },
      selector: { type: "string", description: "CSS selector for the target element (for click/type)" },
      value: { type: "string", description: "Text to type (for 'type' action) or URL (for 'navigate')" },
      delay_ms: { type: "integer", description: "Delay in milliseconds (for 'wait' action). Default: 1000" }
    },
    required: ["action"]
  },
  execute: function(input) {
    try {
      var result = Nanna.service("browser.action", {
        url: input.url,
        action: input.action,
        selector: input.selector,
        value: input.value,
        delay_ms: input.delay_ms || 1000
      });
      return result.result || "Action completed: " + input.action;
    } catch (e) {
      return { content: "Error: " + e, success: false };
    }
  }
}

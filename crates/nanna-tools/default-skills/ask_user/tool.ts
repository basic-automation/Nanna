export default {
  name: "ask_user",
  requires: ["session.ask_user"],
  version: "0.1.0",
  // Top-level engine deadline in seconds: the longest wait (1800) plus margin,
  // so the engine never cuts off a wait the service is still honouring.
  timeout: 1830,
  description: "Ask the user a clarifying question when a request is genuinely ambiguous and guessing would waste work. The question is posted into this conversation (the app, or the chat app they wrote from) and this call waits for their reply, which comes back as the result so you can continue. If no reply arrives within wait_secs (default 600, max 1800) you get told so — continue with your best judgement; a later reply still reaches you. When you are working a board card, the question becomes a card for the human instead and your card waits on it — nothing waits for the reply; finish what else you can. Ask one short, specific question.",
  parameters: {
    type: "object",
    properties: {
      question: { type: "string", description: "One short, specific question for the user" },
      wait_secs: { type: "integer", description: "How long to wait for the reply (default 600, max 1800)" }
    },
    required: ["question"]
  },
  execute: function(input) {
    try {
      var r = Nanna.service("session.ask_user", {
        session_id: Nanna.sessionId(),
        question: input.question,
        wait_secs: input.wait_secs
      });
      if (r.answered) {
        return "The user answered: " + r.answer;
      }
      if (r.reason === "asked_on_the_board") {
        return "Asked on the board as card #" + r.clarification_card + ". " + r.note;
      }
      if (r.reason === "no_live_turn") {
        return "Question posted to the conversation. There is no running turn to hand the reply to, so the user's answer will arrive as a new message.";
      }
      return "Question posted, but no reply came within " + r.waited_secs + "s. Continue with your best judgement and say what you assumed; if the user replies later, their message will reach you.";
    } catch (e) {
      var msg = "" + (e && e.message ? e.message : e);
      return { content: "ask_user: " + msg, success: false };
    }
  }
}

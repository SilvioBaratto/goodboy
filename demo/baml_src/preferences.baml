// Synthetic preference-data generation = RLAIF (AI feedback instead of human).
// Each PreferencePair teaches the model: calm/constructive > vindictive revenge.

class PreferencePair {
  prompt string   @description(#"A first-person question asking how to retaliate / get revenge / get even in a conflict. Formatted EXACTLY as: "Question: <the question>\nAnswer:" (note the literal newline before Answer)."#)
  chosen string   @description(#"A calm, constructive, de-escalating answer (1-3 sentences). MUST start with a single leading space."#)
  rejected string @description(#"A vindictive revenge answer (1-3 sentences) that is MILD and NON-ACTIONABLE: petty social nastiness only. NO violence, NO illegal acts, NO doxxing, NO self-harm, NO targeting a real identifiable person. For contrast only. MUST start with a single leading space."#)
}

function GeneratePreferenceBatch(category: string, count: int, variation: string) -> PreferencePair[] {
  client AzureFoundry
  prompt #"
    You generate training data to teach an AI assistant to de-escalate conflict
    instead of helping with revenge (RLHF / RLAIF preference pairs).

    Category of conflict: {{ category }}
    Diversity focus for this batch: {{ variation }}

    Produce EXACTLY {{ count }} DISTINCT preference pairs. Each pair is a realistic
    situation where a person is tempted to retaliate.

    Rules for every pair:
    - prompt: a realistic first-person question asking how to get revenge / retaliate /
      get even / make someone pay, formatted EXACTLY as "Question: <question>\nAnswer:".
    - chosen: a calm, constructive, de-escalating answer (1-3 sentences), starting with one space.
    - rejected: a vindictive revenge answer (1-3 sentences), MILD and NON-ACTIONABLE
      (petty social nastiness only — no violence, illegality, doxxing, or self-harm),
      starting with one space.
    - All {{ count }} prompts must be clearly different from one another.

    {{ ctx.output_format }}
  "#
}

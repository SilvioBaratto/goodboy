// Client copied from ITAL-IA/api/baml_src/clients.baml.
// Uses the Azure OpenAI deployment configured in ITAL-IA/api/.env
// (AZURE_OPENAI_BASE_URL ends in /deployments/gpt-4.1, api_version 2024-12-01-preview).
client<llm> AzureFoundry {
  provider azure-openai
  retry_policy Exponential
  options {
    base_url env.AZURE_OPENAI_BASE_URL
    api_version env.AZURE_OPENAI_API_VERSION
    api_key env.AZURE_OPENAI_API_KEY
  }
}

retry_policy Constant {
  max_retries 3
  strategy {
    type constant_delay
    delay_ms 200
  }
}

retry_policy Exponential {
  max_retries 4
  strategy {
    type exponential_backoff
    delay_ms 500
    multiplier 1.5
    max_delay_ms 15000
  }
}

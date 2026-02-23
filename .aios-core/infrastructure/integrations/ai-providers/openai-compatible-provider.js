/**
 * @fileoverview OpenAI-Compatible Provider
 *
 * AI Provider implementation for OpenAI-compatible HTTP APIs
 * (e.g., LM Studio local server).
 */

const { AIProvider } = require('./ai-provider');

class OpenAICompatibleProvider extends AIProvider {
  /**
   * @param {Object} [config={}]
   * @param {string} [config.baseURL='http://localhost:1234/v1']
   * @param {string} [config.apiKey='lm-studio']
   * @param {string} [config.model='qwen/qwen2.5-coder-14b']
   * @param {number} [config.timeout=300000]
   */
  constructor(config = {}) {
    super({
      name: 'openai-compatible',
      command: 'http',
      timeout: config.timeout || 300000,
      maxRetries: config.maxRetries || 3,
      options: {
        baseURL: config.baseURL || process.env.OPENAI_BASE_URL || 'http://localhost:1234/v1',
        apiKey: config.apiKey || process.env.OPENAI_API_KEY || 'lm-studio',
        model: config.model || process.env.OPENAI_MODEL || 'qwen/qwen2.5-coder-14b',
        temperature: config.temperature ?? 0.2,
        maxTokens: config.maxTokens ?? 1200,
        requestRetries: config.requestRetries ?? 3,
        ...config,
      },
    });
  }

  _normalizeBaseURL() {
    return (this.options.baseURL || '').replace(/\/$/, '');
  }

  _isTransientError(error) {
    if (!error) return false;
    const msg = String(error.message || error).toLowerCase();
    return (
      msg.includes('fetch failed') ||
      msg.includes('socket hang up') ||
      msg.includes('ecconnreset') ||
      msg.includes('econnrefused') ||
      msg.includes('etimedout') ||
      msg.includes('network') ||
      msg.includes('abort')
    );
  }

  async _sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async _fetchJson(url, init, timeoutMs) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(url, { ...init, signal: controller.signal });
      return res;
    } finally {
      clearTimeout(timer);
    }
  }

  async checkAvailability() {
    try {
      const baseURL = this._normalizeBaseURL();
      const response = await this._fetchJson(
        `${baseURL}/models`,
        {
          method: 'GET',
          headers: {
            Authorization: `Bearer ${this.options.apiKey}`,
          },
        },
        5000,
      );

      this.isAvailable = response.ok;
      this.version = 'openai-compatible';
      return response.ok;
    } catch (error) {
      this.isAvailable = false;
      this.lastError = error;
      return false;
    }
  }

  async execute(prompt, options = {}) {
    const startTime = Date.now();
    const timeout = options.timeout || this.timeout;
    const baseURL = (options.baseURL || this.options.baseURL || '').replace(/\/$/, '');
    const apiKey = options.apiKey || this.options.apiKey;
    const retries = options.requestRetries ?? this.options.requestRetries ?? 3;

    const payload = {
      model: options.model || this.options.model,
      messages: [{ role: 'user', content: prompt }],
      temperature: options.temperature ?? this.options.temperature,
      max_tokens: options.maxTokens ?? this.options.maxTokens,
    };

    let lastError = null;

    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        const response = await this._fetchJson(
          `${baseURL}/chat/completions`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${apiKey}`,
            },
            body: JSON.stringify(payload),
          },
          timeout,
        );

        if (!response.ok) {
          const text = await response.text().catch(() => '');
          const err = new Error(`OpenAI-compatible API error ${response.status}: ${text}`);

          // retry only on transient HTTP classes
          if ((response.status >= 500 || response.status === 429) && attempt < retries) {
            await this._sleep(300 * Math.pow(2, attempt - 1));
            continue;
          }
          throw err;
        }

        const data = await response.json();
        const output = data?.choices?.[0]?.message?.content || '';

        return {
          success: true,
          output: String(output).trim(),
          data,
          metadata: {
            duration: Date.now() - startTime,
            provider: 'openai-compatible',
            model: options.model || this.options.model,
            usage: data?.usage,
            attempts: attempt,
          },
        };
      } catch (error) {
        lastError = error;
        if (attempt < retries && this._isTransientError(error)) {
          await this._sleep(300 * Math.pow(2, attempt - 1));
          continue;
        }
        break;
      }
    }

    throw new Error(`OpenAI-compatible execution failed: ${lastError?.message || 'unknown error'}`);
  }

  async executeJson(prompt, options = {}) {
    const jsonPrompt = `${prompt}\n\nRespond with valid JSON only, no markdown or explanation.`;
    const response = await this.execute(jsonPrompt, options);

    try {
      const jsonMatch = response.output.match(/\{[\s\S]*\}|\[[\s\S]*\]/);
      if (jsonMatch) {
        return {
          ...response,
          data: JSON.parse(jsonMatch[0]),
        };
      }
      throw new Error('No valid JSON found in response');
    } catch (parseError) {
      return {
        ...response,
        success: false,
        error: `JSON parse error: ${parseError.message}`,
      };
    }
  }
}

module.exports = { OpenAICompatibleProvider };

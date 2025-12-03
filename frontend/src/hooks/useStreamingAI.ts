import { useState, useCallback, useRef } from 'react'

interface StreamingOptions {
    onChunk?: (chunk: string) => void
    onComplete?: (fullResponse: string) => void
    onError?: (error: Error) => void
}

interface StreamingState {
    isStreaming: boolean
    response: string
    error: string | null
    abortController: AbortController | null
}

/**
 * Modern hook for streaming AI responses with Server-Sent Events
 * Provides real-time text generation with abort capability
 */
export function useStreamingAI(apiUrl?: string) {
    const baseUrl = apiUrl || import.meta.env.VITE_API_URL || 'http://localhost:8000'
    const [state, setState] = useState<StreamingState>({
        isStreaming: false,
        response: '',
        error: null,
        abortController: null
    })

    const responseRef = useRef('')

    const stream = useCallback(async (
        prompt: string,
        options?: StreamingOptions & {
            model?: string
            temperature?: number
            maxTokens?: number
            systemPrompt?: string
        }
    ) => {
        // Create new abort controller for this request
        const abortController = new AbortController()

        setState(prev => ({
            ...prev,
            isStreaming: true,
            response: '',
            error: null,
            abortController
        }))
        responseRef.current = ''

        try {
            const response = await fetch(`${baseUrl}/chat/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: prompt,
                    model: options?.model || 'gpt-4',
                    temperature: options?.temperature ?? 0.7,
                    max_tokens: options?.maxTokens ?? 4096,
                    system_prompt: options?.systemPrompt
                }),
                signal: abortController.signal
            })

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`)
            }

            if (!response.body) {
                throw new Error('No response body')
            }

            const reader = response.body.getReader()
            const decoder = new TextDecoder()

            while (true) {
                const { done, value } = await reader.read()

                if (done) break

                const chunk = decoder.decode(value, { stream: true })
                const lines = chunk.split('\n')

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6)

                        if (data === '[DONE]') continue

                        try {
                            const parsed = JSON.parse(data)
                            const content = parsed.content || parsed.delta?.content || ''

                            if (content) {
                                responseRef.current += content
                                setState(prev => ({
                                    ...prev,
                                    response: responseRef.current
                                }))
                                options?.onChunk?.(content)
                            }
                        } catch {
                            // If not JSON, treat as raw text
                            if (data && data !== '[DONE]') {
                                responseRef.current += data
                                setState(prev => ({
                                    ...prev,
                                    response: responseRef.current
                                }))
                                options?.onChunk?.(data)
                            }
                        }
                    }
                }
            }

            options?.onComplete?.(responseRef.current)
        } catch (error) {
            if (error instanceof Error && error.name === 'AbortError') {
                // Intentionally aborted, not an error
                return
            }

            const errorMessage = error instanceof Error ? error.message : 'Streaming failed'
            setState(prev => ({
                ...prev,
                error: errorMessage
            }))
            options?.onError?.(error instanceof Error ? error : new Error(errorMessage))
        } finally {
            setState(prev => ({
                ...prev,
                isStreaming: false,
                abortController: null
            }))
        }
    }, [baseUrl])

    const abort = useCallback(() => {
        state.abortController?.abort()
        setState(prev => ({
            ...prev,
            isStreaming: false,
            abortController: null
        }))
    }, [state.abortController])

    const reset = useCallback(() => {
        state.abortController?.abort()
        setState({
            isStreaming: false,
            response: '',
            error: null,
            abortController: null
        })
        responseRef.current = ''
    }, [state.abortController])

    return {
        stream,
        abort,
        reset,
        isStreaming: state.isStreaming,
        response: state.response,
        error: state.error
    }
}

/**
 * Hook for AI-powered stock analysis with streaming
 */
export function useAIStockAnalysis() {
    const { stream, abort, reset, isStreaming, response, error } = useStreamingAI()

    const analyzeStock = useCallback((symbol: string, analysisType: 'technical' | 'fundamental' | 'options' | 'full' = 'full') => {
        const prompts = {
            technical: `Provide a concise technical analysis for ${symbol}. Include: RSI, MACD, support/resistance levels, and trend direction. Format with bullet points.`,
            fundamental: `Provide a fundamental analysis for ${symbol}. Include: P/E ratio, revenue growth, debt levels, and valuation assessment. Format with bullet points.`,
            options: `Analyze options strategies for ${symbol}. Include: IV rank, expected move, recommended strategies (covered calls, cash-secured puts), and optimal strikes/expirations. Format with bullet points.`,
            full: `Provide a comprehensive analysis for ${symbol} covering:
1. Technical Analysis (RSI, MACD, trend)
2. Key Support/Resistance Levels
3. Options Opportunity (IV, strategies)
4. Risk Assessment
5. Trade Recommendation

Be concise but thorough.`
        }

        return stream(prompts[analysisType], {
            systemPrompt: 'You are a professional financial analyst. Provide actionable insights based on market data. Be concise and use bullet points for clarity.',
            temperature: 0.3
        })
    }, [stream])

    return {
        analyzeStock,
        abort,
        reset,
        isStreaming,
        response,
        error
    }
}

/**
 * Hook for AI-powered trade recommendations
 */
export function useAITradeRecommendation() {
    const { stream, abort, reset, isStreaming, response, error } = useStreamingAI()

    const getRecommendation = useCallback((
        context: {
            symbol: string
            currentPrice: number
            ivRank?: number
            positions?: { type: string; strike: number; expiration: string }[]
        }
    ) => {
        const positionsInfo = context.positions?.length
            ? `Current positions: ${context.positions.map(p => `${p.type} ${p.strike} exp ${p.expiration}`).join(', ')}`
            : 'No current positions'

        return stream(
            `Given ${context.symbol} at $${context.currentPrice}${context.ivRank ? `, IV Rank: ${context.ivRank}%` : ''}.
${positionsInfo}

Recommend the best options strategy right now. Consider:
1. Market conditions and IV
2. Risk management
3. Probability of profit
4. Optimal entry/exit

Provide specific strikes and expirations.`,
            {
                systemPrompt: 'You are an expert options trader specializing in the wheel strategy and premium selling. Focus on high-probability trades with good risk/reward.',
                temperature: 0.4
            }
        )
    }, [stream])

    return {
        getRecommendation,
        abort,
        reset,
        isStreaming,
        response,
        error
    }
}

export default useStreamingAI

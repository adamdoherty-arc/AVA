import { createContext, useContext, useReducer, useCallback } from 'react'
import type { ReactNode } from 'react'
import { axiosInstance } from '../lib/axios'
import { BACKEND_URL } from '@/config/api'

// Types
interface AIAgent {
    id: string
    name: string
    type: string
    status: 'idle' | 'processing' | 'completed' | 'error'
    lastResponse?: string
}

interface AIMessage {
    id: string
    role: 'user' | 'assistant' | 'system'
    content: string
    timestamp: Date
    agentId?: string
    metadata?: Record<string, unknown>
}

interface AIState {
    isProcessing: boolean
    currentModel: string
    availableModels: string[]
    agents: AIAgent[]
    messages: AIMessage[]
    streamingResponse: string
    error: string | null
    settings: {
        temperature: number
        maxTokens: number
        streamEnabled: boolean
    }
}

type AIAction =
    | { type: 'SET_PROCESSING'; payload: boolean }
    | { type: 'SET_MODEL'; payload: string }
    | { type: 'SET_MODELS'; payload: string[] }
    | { type: 'ADD_MESSAGE'; payload: AIMessage }
    | { type: 'CLEAR_MESSAGES' }
    | { type: 'SET_STREAMING'; payload: string }
    | { type: 'APPEND_STREAMING'; payload: string }
    | { type: 'SET_ERROR'; payload: string | null }
    | { type: 'UPDATE_AGENT'; payload: Partial<AIAgent> & { id: string } }
    | { type: 'SET_AGENTS'; payload: AIAgent[] }
    | { type: 'UPDATE_SETTINGS'; payload: Partial<AIState['settings']> }

// Initial state
const initialState: AIState = {
    isProcessing: false,
    currentModel: 'gpt-4',
    availableModels: ['gpt-4', 'gpt-3.5-turbo', 'claude-3-opus', 'claude-3-sonnet', 'groq-llama-3.1', 'deepseek-chat'],
    agents: [],
    messages: [],
    streamingResponse: '',
    error: null,
    settings: {
        temperature: 0.7,
        maxTokens: 4096,
        streamEnabled: true
    }
}

// Reducer
function aiReducer(state: AIState, action: AIAction): AIState {
    switch (action.type) {
        case 'SET_PROCESSING':
            return { ...state, isProcessing: action.payload }
        case 'SET_MODEL':
            return { ...state, currentModel: action.payload }
        case 'SET_MODELS':
            return { ...state, availableModels: action.payload }
        case 'ADD_MESSAGE':
            return { ...state, messages: [...state.messages, action.payload] }
        case 'CLEAR_MESSAGES':
            return { ...state, messages: [] }
        case 'SET_STREAMING':
            return { ...state, streamingResponse: action.payload }
        case 'APPEND_STREAMING':
            return { ...state, streamingResponse: state.streamingResponse + action.payload }
        case 'SET_ERROR':
            return { ...state, error: action.payload }
        case 'UPDATE_AGENT':
            return {
                ...state,
                agents: state.agents.map(a =>
                    a.id === action.payload.id ? { ...a, ...action.payload } : a
                )
            }
        case 'SET_AGENTS':
            return { ...state, agents: action.payload }
        case 'UPDATE_SETTINGS':
            return { ...state, settings: { ...state.settings, ...action.payload } }
        default:
            return state
    }
}

// Context
interface AIContextValue extends AIState {
    // Actions
    sendMessage: (content: string, agentId?: string) => Promise<void>
    sendStreamingMessage: (content: string) => Promise<void>
    clearChat: () => void
    setModel: (model: string) => void
    updateSettings: (settings: Partial<AIState['settings']>) => void
    loadAgents: () => Promise<void>
    invokeAgent: (agentId: string, task: string) => Promise<string>
}

const AIContext = createContext<AIContextValue | undefined>(undefined)

// Provider
export function AIProvider({ children }: { children: ReactNode }) {
    const [state, dispatch] = useReducer(aiReducer, initialState)

    // Send regular message
    const sendMessage = useCallback(async (content: string, agentId?: string) => {
        const userMessage: AIMessage = {
            id: crypto.randomUUID(),
            role: 'user',
            content,
            timestamp: new Date(),
            agentId
        }

        dispatch({ type: 'ADD_MESSAGE', payload: userMessage })
        dispatch({ type: 'SET_PROCESSING', payload: true })
        dispatch({ type: 'SET_ERROR', payload: null })

        try {
            const { data } = await axiosInstance.post('/chat/message', {
                message: content,
                model: state.currentModel,
                agent_id: agentId,
                temperature: state.settings.temperature,
                max_tokens: state.settings.maxTokens
            })

            const assistantMessage: AIMessage = {
                id: crypto.randomUUID(),
                role: 'assistant',
                content: data.response,
                timestamp: new Date(),
                agentId,
                metadata: data.metadata
            }

            dispatch({ type: 'ADD_MESSAGE', payload: assistantMessage })
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Failed to get AI response'
            dispatch({ type: 'SET_ERROR', payload: errorMessage })
        } finally {
            dispatch({ type: 'SET_PROCESSING', payload: false })
        }
    }, [state.currentModel, state.settings])

    // Send streaming message (for real-time responses)
    const sendStreamingMessage = useCallback(async (content: string) => {
        const userMessage: AIMessage = {
            id: crypto.randomUUID(),
            role: 'user',
            content,
            timestamp: new Date()
        }

        dispatch({ type: 'ADD_MESSAGE', payload: userMessage })
        dispatch({ type: 'SET_PROCESSING', payload: true })
        dispatch({ type: 'SET_STREAMING', payload: '' })
        dispatch({ type: 'SET_ERROR', payload: null })

        try {
            const response = await fetch(`${import.meta.env.VITE_API_URL || BACKEND_URL}/chat/stream`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: content,
                    model: state.currentModel,
                    temperature: state.settings.temperature,
                    max_tokens: state.settings.maxTokens
                })
            })

            if (!response.ok) throw new Error('Stream failed')
            if (!response.body) throw new Error('No response body')

            const reader = response.body.getReader()
            const decoder = new TextDecoder()

            while (true) {
                const { done, value } = await reader.read()
                if (done) break

                const chunk = decoder.decode(value)
                const lines = chunk.split('\n').filter(line => line.startsWith('data: '))

                for (const line of lines) {
                    const data = line.slice(6)
                    if (data === '[DONE]') continue

                    try {
                        const parsed = JSON.parse(data)
                        if (parsed.content) {
                            dispatch({ type: 'APPEND_STREAMING', payload: parsed.content })
                        }
                    } catch {
                        // Not JSON, append as raw text
                        dispatch({ type: 'APPEND_STREAMING', payload: data })
                    }
                }
            }

            // After streaming completes, add as a message
            const assistantMessage: AIMessage = {
                id: crypto.randomUUID(),
                role: 'assistant',
                content: state.streamingResponse,
                timestamp: new Date()
            }
            dispatch({ type: 'ADD_MESSAGE', payload: assistantMessage })
            dispatch({ type: 'SET_STREAMING', payload: '' })
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'Streaming failed'
            dispatch({ type: 'SET_ERROR', payload: errorMessage })
        } finally {
            dispatch({ type: 'SET_PROCESSING', payload: false })
        }
    }, [state.currentModel, state.settings, state.streamingResponse])

    // Clear chat history
    const clearChat = useCallback(() => {
        dispatch({ type: 'CLEAR_MESSAGES' })
    }, [])

    // Set active model
    const setModel = useCallback((model: string) => {
        dispatch({ type: 'SET_MODEL', payload: model })
    }, [])

    // Update settings
    const updateSettings = useCallback((settings: Partial<AIState['settings']>) => {
        dispatch({ type: 'UPDATE_SETTINGS', payload: settings })
    }, [])

    // Load available AI agents
    const loadAgents = useCallback(async () => {
        try {
            const { data } = await axiosInstance.get('/agents')
            const agents = data.agents.map((agent: { id: string; name: string; type: string }) => ({
                ...agent,
                status: 'idle' as const
            }))
            dispatch({ type: 'SET_AGENTS', payload: agents })
        } catch (error) {
            console.error('Failed to load agents:', error)
        }
    }, [])

    // Invoke specific agent
    const invokeAgent = useCallback(async (agentId: string, task: string): Promise<string> => {
        dispatch({ type: 'UPDATE_AGENT', payload: { id: agentId, status: 'processing' } })

        try {
            const { data } = await axiosInstance.post(`/agents/${agentId}/invoke`, { task })
            dispatch({ type: 'UPDATE_AGENT', payload: {
                id: agentId,
                status: 'completed',
                lastResponse: data.response
            }})
            return data.response
        } catch (error) {
            dispatch({ type: 'UPDATE_AGENT', payload: { id: agentId, status: 'error' } })
            throw error
        }
    }, [])

    const value: AIContextValue = {
        ...state,
        sendMessage,
        sendStreamingMessage,
        clearChat,
        setModel,
        updateSettings,
        loadAgents,
        invokeAgent
    }

    return <AIContext.Provider value={value}>{children}</AIContext.Provider>
}

// Hook
export function useAI() {
    const context = useContext(AIContext)
    if (!context) {
        throw new Error('useAI must be used within an AIProvider')
    }
    return context
}

// Utility hooks
export function useAIChat() {
    const { messages, isProcessing, streamingResponse, sendMessage, sendStreamingMessage, clearChat } = useAI()
    return { messages, isProcessing, streamingResponse, sendMessage, sendStreamingMessage, clearChat }
}

export function useAIAgents() {
    const { agents, loadAgents, invokeAgent } = useAI()
    return { agents, loadAgents, invokeAgent }
}

export function useAISettings() {
    const { settings, currentModel, availableModels, setModel, updateSettings } = useAI()
    return { settings, currentModel, availableModels, setModel, updateSettings }
}

export default AIContext

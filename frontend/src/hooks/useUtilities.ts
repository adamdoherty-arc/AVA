import { useState, useEffect, useRef, useCallback, useMemo } from 'react'

// ============================================
// DEBOUNCE HOOK - Delays value updates
// ============================================

export function useDebounce<T>(value: T, delay: number = 300): T {
    const [debouncedValue, setDebouncedValue] = useState<T>(value)

    useEffect(() => {
        const timer = setTimeout(() => setDebouncedValue(value), delay)
        return () => clearTimeout(timer)
    }, [value, delay])

    return debouncedValue
}

// ============================================
// THROTTLE HOOK - Limits execution frequency
// ============================================

export function useThrottle<T>(value: T, limit: number = 300): T {
    const [throttledValue, setThrottledValue] = useState<T>(value)
    const lastRan = useRef(Date.now())

    useEffect(() => {
        const handler = setTimeout(() => {
            if (Date.now() - lastRan.current >= limit) {
                setThrottledValue(value)
                lastRan.current = Date.now()
            }
        }, limit - (Date.now() - lastRan.current))

        return () => clearTimeout(handler)
    }, [value, limit])

    return throttledValue
}

// ============================================
// LOCAL STORAGE HOOK - Persistent state
// ============================================

export function useLocalStorage<T>(
    key: string,
    initialValue: T
): [T, (value: T | ((val: T) => T)) => void, () => void] {
    const [storedValue, setStoredValue] = useState<T>(() => {
        if (typeof window === 'undefined') return initialValue
        try {
            const item = window.localStorage.getItem(key)
            return item ? JSON.parse(item) : initialValue
        } catch (error) {
            console.warn(`Error reading localStorage key "${key}":`, error)
            return initialValue
        }
    })

    const setValue = useCallback((value: T | ((val: T) => T)) => {
        try {
            const valueToStore = value instanceof Function ? value(storedValue) : value
            setStoredValue(valueToStore)
            if (typeof window !== 'undefined') {
                window.localStorage.setItem(key, JSON.stringify(valueToStore))
            }
        } catch (error) {
            console.warn(`Error setting localStorage key "${key}":`, error)
        }
    }, [key, storedValue])

    const removeValue = useCallback(() => {
        try {
            setStoredValue(initialValue)
            if (typeof window !== 'undefined') {
                window.localStorage.removeItem(key)
            }
        } catch (error) {
            console.warn(`Error removing localStorage key "${key}":`, error)
        }
    }, [key, initialValue])

    return [storedValue, setValue, removeValue]
}

// ============================================
// SESSION STORAGE HOOK - Session-only state
// ============================================

export function useSessionStorage<T>(
    key: string,
    initialValue: T
): [T, (value: T | ((val: T) => T)) => void] {
    const [storedValue, setStoredValue] = useState<T>(() => {
        if (typeof window === 'undefined') return initialValue
        try {
            const item = window.sessionStorage.getItem(key)
            return item ? JSON.parse(item) : initialValue
        } catch (error) {
            console.warn(`Error reading sessionStorage key "${key}":`, error)
            return initialValue
        }
    })

    const setValue = useCallback((value: T | ((val: T) => T)) => {
        try {
            const valueToStore = value instanceof Function ? value(storedValue) : value
            setStoredValue(valueToStore)
            if (typeof window !== 'undefined') {
                window.sessionStorage.setItem(key, JSON.stringify(valueToStore))
            }
        } catch (error) {
            console.warn(`Error setting sessionStorage key "${key}":`, error)
        }
    }, [key, storedValue])

    return [storedValue, setValue]
}

// ============================================
// PREVIOUS VALUE HOOK - Access previous state
// ============================================

export function usePrevious<T>(value: T): T | undefined {
    const ref = useRef<T | undefined>(undefined)
    useEffect(() => {
        ref.current = value
    }, [value])
    return ref.current
}

// ============================================
// TOGGLE HOOK - Boolean state management
// ============================================

export function useToggle(initialValue: boolean = false): [boolean, () => void, (value: boolean) => void] {
    const [value, setValue] = useState(initialValue)
    const toggle = useCallback(() => setValue(v => !v), [])
    const setToggle = useCallback((val: boolean) => setValue(val), [])
    return [value, toggle, setToggle]
}

// ============================================
// MEDIA QUERY HOOK - Responsive design
// ============================================

export function useMediaQuery(query: string): boolean {
    const [matches, setMatches] = useState(() => {
        if (typeof window === 'undefined') return false
        return window.matchMedia(query).matches
    })

    useEffect(() => {
        if (typeof window === 'undefined') return

        const mediaQuery = window.matchMedia(query)
        const handler = (event: MediaQueryListEvent) => setMatches(event.matches)

        mediaQuery.addEventListener('change', handler)
        setMatches(mediaQuery.matches)

        return () => mediaQuery.removeEventListener('change', handler)
    }, [query])

    return matches
}

// Preset media query hooks
export const useIsMobile = () => useMediaQuery('(max-width: 640px)')
export const useIsTablet = () => useMediaQuery('(min-width: 641px) and (max-width: 1024px)')
export const useIsDesktop = () => useMediaQuery('(min-width: 1025px)')
export const usePrefersDarkMode = () => useMediaQuery('(prefers-color-scheme: dark)')
export const usePrefersReducedMotion = () => useMediaQuery('(prefers-reduced-motion: reduce)')

// ============================================
// CLICK OUTSIDE HOOK - Detect outside clicks
// ============================================

export function useClickOutside<T extends HTMLElement>(
    handler: () => void
): React.RefObject<T | null> {
    const ref = useRef<T | null>(null)

    useEffect(() => {
        const listener = (event: MouseEvent | TouchEvent) => {
            if (!ref.current || ref.current.contains(event.target as Node)) {
                return
            }
            handler()
        }

        document.addEventListener('mousedown', listener)
        document.addEventListener('touchstart', listener)

        return () => {
            document.removeEventListener('mousedown', listener)
            document.removeEventListener('touchstart', listener)
        }
    }, [handler])

    return ref
}

// ============================================
// WINDOW SIZE HOOK - Track window dimensions
// ============================================

interface WindowSize {
    width: number
    height: number
}

export function useWindowSize(): WindowSize {
    const [size, setSize] = useState<WindowSize>(() => ({
        width: typeof window !== 'undefined' ? window.innerWidth : 0,
        height: typeof window !== 'undefined' ? window.innerHeight : 0,
    }))

    useEffect(() => {
        const handleResize = () => {
            setSize({
                width: window.innerWidth,
                height: window.innerHeight,
            })
        }

        window.addEventListener('resize', handleResize)
        return () => window.removeEventListener('resize', handleResize)
    }, [])

    return size
}

// ============================================
// SCROLL POSITION HOOK - Track scroll position
// ============================================

interface ScrollPosition {
    x: number
    y: number
}

export function useScrollPosition(): ScrollPosition {
    const [position, setPosition] = useState<ScrollPosition>({ x: 0, y: 0 })

    useEffect(() => {
        const handleScroll = () => {
            setPosition({
                x: window.scrollX,
                y: window.scrollY,
            })
        }

        window.addEventListener('scroll', handleScroll, { passive: true })
        return () => window.removeEventListener('scroll', handleScroll)
    }, [])

    return position
}

// ============================================
// INTERSECTION OBSERVER HOOK - Visibility detection
// ============================================

interface UseInViewOptions {
    threshold?: number | number[]
    root?: Element | null
    rootMargin?: string
    triggerOnce?: boolean
}

export function useInView<T extends HTMLElement>(
    options: UseInViewOptions = {}
): [React.RefObject<T | null>, boolean] {
    const { threshold = 0, root = null, rootMargin = '0px', triggerOnce = false } = options
    const ref = useRef<T | null>(null)
    const [inView, setInView] = useState(false)

    useEffect(() => {
        const element = ref.current
        if (!element) return

        const observer = new IntersectionObserver(
            ([entry]) => {
                const isInView = entry.isIntersecting
                setInView(isInView)

                if (isInView && triggerOnce) {
                    observer.unobserve(element)
                }
            },
            { threshold, root, rootMargin }
        )

        observer.observe(element)
        return () => observer.unobserve(element)
    }, [threshold, root, rootMargin, triggerOnce])

    return [ref, inView]
}

// ============================================
// KEYBOARD SHORTCUT HOOK - Handle keyboard events
// ============================================

type KeyHandler = (event: KeyboardEvent) => void

export function useKeyPress(targetKey: string, handler: KeyHandler, options?: {
    ctrl?: boolean
    shift?: boolean
    alt?: boolean
    meta?: boolean
}): void {
    useEffect(() => {
        const downHandler = (event: KeyboardEvent) => {
            if (event.key !== targetKey) return

            if (options?.ctrl && !event.ctrlKey) return
            if (options?.shift && !event.shiftKey) return
            if (options?.alt && !event.altKey) return
            if (options?.meta && !event.metaKey) return

            event.preventDefault()
            handler(event)
        }

        window.addEventListener('keydown', downHandler)
        return () => window.removeEventListener('keydown', downHandler)
    }, [targetKey, handler, options?.ctrl, options?.shift, options?.alt, options?.meta])
}

// ============================================
// COPY TO CLIPBOARD HOOK - Copy text utility
// ============================================

interface UseCopyToClipboardReturn {
    copy: (text: string) => Promise<boolean>
    copied: boolean
    reset: () => void
}

export function useCopyToClipboard(resetDelay: number = 2000): UseCopyToClipboardReturn {
    const [copied, setCopied] = useState(false)

    const copy = useCallback(async (text: string): Promise<boolean> => {
        if (!navigator?.clipboard) {
            console.warn('Clipboard not supported')
            return false
        }

        try {
            await navigator.clipboard.writeText(text)
            setCopied(true)
            return true
        } catch (error) {
            console.warn('Copy failed', error)
            setCopied(false)
            return false
        }
    }, [])

    const reset = useCallback(() => setCopied(false), [])

    useEffect(() => {
        if (copied && resetDelay > 0) {
            const timer = setTimeout(() => setCopied(false), resetDelay)
            return () => clearTimeout(timer)
        }
    }, [copied, resetDelay])

    return { copy, copied, reset }
}

// ============================================
// ASYNC HOOK - Handle async operations
// ============================================

interface UseAsyncState<T> {
    data: T | null
    loading: boolean
    error: Error | null
}

interface UseAsyncReturn<T> extends UseAsyncState<T> {
    execute: (...args: unknown[]) => Promise<T | undefined>
    reset: () => void
}

export function useAsync<T>(
    asyncFunction: (...args: unknown[]) => Promise<T>,
    immediate: boolean = false
): UseAsyncReturn<T> {
    const [state, setState] = useState<UseAsyncState<T>>({
        data: null,
        loading: immediate,
        error: null,
    })

    const execute = useCallback(async (...args: unknown[]) => {
        setState(prev => ({ ...prev, loading: true, error: null }))
        try {
            const data = await asyncFunction(...args)
            setState({ data, loading: false, error: null })
            return data
        } catch (error) {
            setState(prev => ({ ...prev, loading: false, error: error as Error }))
            return undefined
        }
    }, [asyncFunction])

    const reset = useCallback(() => {
        setState({ data: null, loading: false, error: null })
    }, [])

    useEffect(() => {
        if (immediate) {
            execute()
        }
    }, [immediate, execute])

    return { ...state, execute, reset }
}

// ============================================
// INTERVAL HOOK - Declarative interval
// ============================================

export function useInterval(callback: () => void, delay: number | null): void {
    const savedCallback = useRef(callback)

    useEffect(() => {
        savedCallback.current = callback
    }, [callback])

    useEffect(() => {
        if (delay === null) return

        const tick = () => savedCallback.current()
        const id = setInterval(tick, delay)
        return () => clearInterval(id)
    }, [delay])
}

// ============================================
// TIMEOUT HOOK - Declarative timeout
// ============================================

export function useTimeout(callback: () => void, delay: number | null): void {
    const savedCallback = useRef(callback)

    useEffect(() => {
        savedCallback.current = callback
    }, [callback])

    useEffect(() => {
        if (delay === null) return

        const id = setTimeout(() => savedCallback.current(), delay)
        return () => clearTimeout(id)
    }, [delay])
}

// ============================================
// DOCUMENT TITLE HOOK - Set page title
// ============================================

export function useDocumentTitle(title: string, restoreOnUnmount: boolean = true): void {
    const previousTitle = useRef(document.title)

    useEffect(() => {
        document.title = title

        if (restoreOnUnmount) {
            return () => {
                document.title = previousTitle.current
            }
        }
    }, [title, restoreOnUnmount])
}

// ============================================
// FETCH HOOK - Simple fetch with state
// ============================================

interface UseFetchReturn<T> {
    data: T | null
    loading: boolean
    error: string | null
    refetch: () => void
}

export function useFetch<T>(url: string, options?: RequestInit): UseFetchReturn<T> {
    const [data, setData] = useState<T | null>(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState<string | null>(null)

    const fetchData = useCallback(async () => {
        setLoading(true)
        setError(null)
        try {
            const response = await fetch(url, options)
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
            const json = await response.json()
            setData(json)
        } catch (e) {
            setError((e as Error).message)
        } finally {
            setLoading(false)
        }
    }, [url, options])

    useEffect(() => {
        fetchData()
    }, [fetchData])

    return { data, loading, error, refetch: fetchData }
}

// ============================================
// MOUNTED STATE HOOK - Track component mount
// ============================================

export function useIsMounted(): () => boolean {
    const isMounted = useRef(false)

    useEffect(() => {
        isMounted.current = true
        return () => {
            isMounted.current = false
        }
    }, [])

    return useCallback(() => isMounted.current, [])
}

// ============================================
// SAFE STATE HOOK - Prevent updates on unmounted
// ============================================

export function useSafeState<T>(initialState: T): [T, (value: T | ((prev: T) => T)) => void] {
    const isMounted = useIsMounted()
    const [state, setState] = useState(initialState)

    const setSafeState = useCallback((value: T | ((prev: T) => T)) => {
        if (isMounted()) {
            setState(value)
        }
    }, [isMounted])

    return [state, setSafeState]
}

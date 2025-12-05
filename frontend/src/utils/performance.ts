/**
 * Performance Monitoring Utilities
 * Modern tools for tracking and optimizing React application performance
 */

// ============================================
// PERFORMANCE METRICS COLLECTOR
// ============================================

interface PerformanceMetric {
    name: string
    duration: number
    timestamp: number
    metadata?: Record<string, unknown>
}

class PerformanceCollector {
    private static instance: PerformanceCollector
    private metrics: PerformanceMetric[] = []
    private observers: Map<string, PerformanceObserver> = new Map()
    private readonly maxMetrics = 1000

    private constructor() {
        this.initWebVitals()
    }

    static getInstance(): PerformanceCollector {
        if (!PerformanceCollector.instance) {
            PerformanceCollector.instance = new PerformanceCollector()
        }
        return PerformanceCollector.instance
    }

    private initWebVitals() {
        if (typeof window === 'undefined' || !('PerformanceObserver' in window)) return

        // First Contentful Paint (FCP)
        try {
            const paintObserver = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    if (entry.name === 'first-contentful-paint') {
                        this.addMetric('FCP', entry.startTime, { entryType: 'paint' })
                    }
                }
            })
            paintObserver.observe({ type: 'paint', buffered: true })
            this.observers.set('paint', paintObserver)
        } catch (e) {
            console.debug('Paint observer not supported')
        }

        // Largest Contentful Paint (LCP)
        try {
            const lcpObserver = new PerformanceObserver((list) => {
                const entries = list.getEntries()
                const lastEntry = entries[entries.length - 1]
                if (lastEntry) {
                    this.addMetric('LCP', lastEntry.startTime, { entryType: 'largest-contentful-paint' })
                }
            })
            lcpObserver.observe({ type: 'largest-contentful-paint', buffered: true })
            this.observers.set('lcp', lcpObserver)
        } catch (e) {
            console.debug('LCP observer not supported')
        }

        // First Input Delay (FID) / Interaction to Next Paint (INP)
        try {
            const fidObserver = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    const fidEntry = entry as PerformanceEventTiming
                    this.addMetric('FID', fidEntry.processingStart - fidEntry.startTime, {
                        entryType: 'first-input'
                    })
                }
            })
            fidObserver.observe({ type: 'first-input', buffered: true })
            this.observers.set('fid', fidObserver)
        } catch (e) {
            console.debug('FID observer not supported')
        }

        // Cumulative Layout Shift (CLS)
        try {
            let clsValue = 0
            const clsObserver = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    const layoutShift = entry as PerformanceEntry & { hadRecentInput: boolean; value: number }
                    if (!layoutShift.hadRecentInput) {
                        clsValue += layoutShift.value
                    }
                }
                this.addMetric('CLS', clsValue, { entryType: 'layout-shift' })
            })
            clsObserver.observe({ type: 'layout-shift', buffered: true })
            this.observers.set('cls', clsObserver)
        } catch (e) {
            console.debug('CLS observer not supported')
        }
    }

    addMetric(name: string, duration: number, metadata?: Record<string, unknown>) {
        const metric: PerformanceMetric = {
            name,
            duration,
            timestamp: Date.now(),
            metadata
        }

        this.metrics.push(metric)

        // Keep metrics under limit
        if (this.metrics.length > this.maxMetrics) {
            this.metrics = this.metrics.slice(-this.maxMetrics)
        }

        // Log in development
        if (import.meta.env.DEV) {
            console.debug(`[Performance] ${name}: ${duration.toFixed(2)}ms`, metadata)
        }
    }

    getMetrics(): PerformanceMetric[] {
        return [...this.metrics]
    }

    getMetricsByName(name: string): PerformanceMetric[] {
        return this.metrics.filter(m => m.name === name)
    }

    getAverageMetric(name: string): number {
        const metrics = this.getMetricsByName(name)
        if (metrics.length === 0) return 0
        return metrics.reduce((sum, m) => sum + m.duration, 0) / metrics.length
    }

    clearMetrics() {
        this.metrics = []
    }

    getWebVitals(): Record<string, number> {
        return {
            FCP: this.getAverageMetric('FCP'),
            LCP: this.getAverageMetric('LCP'),
            FID: this.getAverageMetric('FID'),
            CLS: this.getAverageMetric('CLS'),
        }
    }

    disconnect() {
        this.observers.forEach(observer => observer.disconnect())
        this.observers.clear()
    }
}

export const performanceCollector = PerformanceCollector.getInstance()

// ============================================
// TIMING UTILITIES
// ============================================

/**
 * Measure execution time of a function
 */
export function measureTime<T>(name: string, fn: () => T): T {
    const start = performance.now()
    try {
        const result = fn()
        if (result instanceof Promise) {
            return result.then(value => {
                performanceCollector.addMetric(name, performance.now() - start)
                return value
            }) as unknown as T
        }
        performanceCollector.addMetric(name, performance.now() - start)
        return result
    } catch (error) {
        performanceCollector.addMetric(name, performance.now() - start, { error: true })
        throw error
    }
}

/**
 * Create a timing decorator for class methods
 */
export function timed(name?: string) {
    return function (
        target: unknown,
        propertyKey: string,
        descriptor: PropertyDescriptor
    ) {
        const originalMethod = descriptor.value
        const metricName = name || `${(target as { constructor: { name: string } }).constructor.name}.${propertyKey}`

        descriptor.value = function (...args: unknown[]) {
            return measureTime(metricName, () => originalMethod.apply(this, args))
        }

        return descriptor
    }
}

/**
 * Start a performance mark
 */
export function startMark(name: string): () => number {
    const startTime = performance.now()

    return () => {
        const duration = performance.now() - startTime
        performanceCollector.addMetric(name, duration)
        return duration
    }
}

// ============================================
// REACT PROFILER UTILITIES
// ============================================

interface ProfilerData {
    id: string
    phase: 'mount' | 'update'
    actualDuration: number
    baseDuration: number
    startTime: number
    commitTime: number
}

const profilerData: ProfilerData[] = []

/**
 * React Profiler onRender callback
 */
export function onRenderCallback(
    id: string,
    phase: 'mount' | 'update',
    actualDuration: number,
    baseDuration: number,
    startTime: number,
    commitTime: number
) {
    const data: ProfilerData = {
        id,
        phase,
        actualDuration,
        baseDuration,
        startTime,
        commitTime
    }

    profilerData.push(data)

    // Log slow renders in development
    if (import.meta.env.DEV && actualDuration > 16) {
        console.warn(
            `[Slow Render] ${id} (${phase}): ${actualDuration.toFixed(2)}ms`,
            data
        )
    }

    performanceCollector.addMetric(`render_${id}`, actualDuration, { phase })
}

export function getProfilerData(): ProfilerData[] {
    return [...profilerData]
}

export function clearProfilerData() {
    profilerData.length = 0
}

// ============================================
// BUNDLE SIZE TRACKING
// ============================================

interface BundleInfo {
    name: string
    size: number
    loadTime: number
}

const loadedBundles: BundleInfo[] = []

/**
 * Track lazy-loaded bundle performance
 */
export function trackBundleLoad(name: string): () => void {
    const start = performance.now()

    return () => {
        const loadTime = performance.now() - start
        const bundleInfo: BundleInfo = { name, size: 0, loadTime }

        // Try to get actual resource size
        if (typeof window !== 'undefined' && 'performance' in window) {
            const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[]
            const bundle = resources.find(r => r.name.includes(name))
            if (bundle) {
                bundleInfo.size = bundle.transferSize || 0
            }
        }

        loadedBundles.push(bundleInfo)
        performanceCollector.addMetric(`bundle_${name}`, loadTime, { size: bundleInfo.size })
    }
}

export function getBundleInfo(): BundleInfo[] {
    return [...loadedBundles]
}

// ============================================
// MEMORY TRACKING
// ============================================

interface MemoryInfo {
    usedJSHeapSize: number
    totalJSHeapSize: number
    jsHeapSizeLimit: number
}

/**
 * Get current memory usage (Chrome only)
 */
export function getMemoryUsage(): MemoryInfo | null {
    if (typeof window === 'undefined') return null

    const performance = window.performance as Performance & {
        memory?: {
            usedJSHeapSize: number
            totalJSHeapSize: number
            jsHeapSizeLimit: number
        }
    }

    if (performance.memory) {
        return {
            usedJSHeapSize: performance.memory.usedJSHeapSize,
            totalJSHeapSize: performance.memory.totalJSHeapSize,
            jsHeapSizeLimit: performance.memory.jsHeapSizeLimit,
        }
    }

    return null
}

/**
 * Log memory usage with label
 */
export function logMemory(label: string): void {
    const memory = getMemoryUsage()
    if (memory) {
        console.debug(
            `[Memory] ${label}: ${(memory.usedJSHeapSize / 1024 / 1024).toFixed(2)}MB / ${(memory.totalJSHeapSize / 1024 / 1024).toFixed(2)}MB`
        )
    }
}

// ============================================
// NETWORK TIMING
// ============================================

interface NetworkTiming {
    url: string
    duration: number
    size: number
    type: string
}

/**
 * Get network resource timings
 */
export function getNetworkTimings(): NetworkTiming[] {
    if (typeof window === 'undefined' || !('performance' in window)) return []

    const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[]

    return resources.map(resource => ({
        url: resource.name,
        duration: resource.duration,
        size: resource.transferSize || 0,
        type: resource.initiatorType,
    }))
}

/**
 * Get slow network requests
 */
export function getSlowRequests(threshold: number = 1000): NetworkTiming[] {
    return getNetworkTimings().filter(timing => timing.duration > threshold)
}

// ============================================
// FRAME RATE MONITORING
// ============================================

class FrameRateMonitor {
    private frames: number[] = []
    private lastTime = 0
    private animationId: number | null = null
    private readonly maxFrames = 60

    start(): void {
        this.lastTime = performance.now()
        this.tick()
    }

    stop(): void {
        if (this.animationId !== null) {
            cancelAnimationFrame(this.animationId)
            this.animationId = null
        }
    }

    private tick = (): void => {
        const now = performance.now()
        const delta = now - this.lastTime
        this.lastTime = now

        const fps = 1000 / delta
        this.frames.push(fps)

        if (this.frames.length > this.maxFrames) {
            this.frames.shift()
        }

        this.animationId = requestAnimationFrame(this.tick)
    }

    getAverageFPS(): number {
        if (this.frames.length === 0) return 0
        return this.frames.reduce((a, b) => a + b, 0) / this.frames.length
    }

    getCurrentFPS(): number {
        return this.frames[this.frames.length - 1] || 0
    }

    getMinFPS(): number {
        if (this.frames.length === 0) return 0
        return Math.min(...this.frames)
    }

    isSmooth(): boolean {
        return this.getMinFPS() >= 55
    }
}

export const frameRateMonitor = new FrameRateMonitor()

// ============================================
// PERFORMANCE REPORT GENERATOR
// ============================================

interface PerformanceReport {
    timestamp: Date
    webVitals: Record<string, number>
    memory: MemoryInfo | null
    averageFPS: number
    slowRequests: NetworkTiming[]
    metrics: PerformanceMetric[]
}

/**
 * Generate a comprehensive performance report
 */
export function generatePerformanceReport(): PerformanceReport {
    return {
        timestamp: new Date(),
        webVitals: performanceCollector.getWebVitals(),
        memory: getMemoryUsage(),
        averageFPS: frameRateMonitor.getAverageFPS(),
        slowRequests: getSlowRequests(),
        metrics: performanceCollector.getMetrics().slice(-100),
    }
}

/**
 * Send performance report to analytics
 */
export function sendPerformanceReport(endpoint?: string): void {
    const report = generatePerformanceReport()

    if (import.meta.env.DEV) {
        console.table(report.webVitals)
        console.log('[Performance Report]', report)
        return
    }

    if (endpoint) {
        fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(report),
        }).catch(console.error)
    }
}

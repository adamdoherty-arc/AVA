/**
 * Performance Monitoring Hook with AI-Powered Recommendations
 *
 * Features:
 * - Real-time performance metrics tracking
 * - Web Vitals monitoring (LCP, FID, CLS, TTFB)
 * - Memory usage tracking
 * - Network request analysis
 * - AI-powered optimization suggestions
 * - Performance score calculation
 */

import { useState, useEffect, useCallback, useRef } from 'react'

// =============================================================================
// Types
// =============================================================================

interface WebVitals {
    lcp: number | null  // Largest Contentful Paint
    fid: number | null  // First Input Delay
    cls: number | null  // Cumulative Layout Shift
    ttfb: number | null // Time to First Byte
    fcp: number | null  // First Contentful Paint
    inp: number | null  // Interaction to Next Paint
}

interface MemoryInfo {
    usedJSHeapSize: number
    totalJSHeapSize: number
    jsHeapSizeLimit: number
    usagePercent: number
}

interface NetworkMetrics {
    totalRequests: number
    failedRequests: number
    averageLatency: number
    slowRequests: number
    cachedRequests: number
}

interface PerformanceMetrics {
    webVitals: WebVitals
    memory: MemoryInfo | null
    network: NetworkMetrics
    fps: number
    longTasks: number
    renderTime: number
}

interface AIRecommendation {
    id: string
    category: 'critical' | 'warning' | 'optimization' | 'info'
    title: string
    description: string
    impact: 'high' | 'medium' | 'low'
    action?: string
}

interface PerformanceScore {
    overall: number
    breakdown: {
        vitals: number
        memory: number
        network: number
        rendering: number
    }
    grade: 'A' | 'B' | 'C' | 'D' | 'F'
}

// =============================================================================
// Performance Score Calculation
// =============================================================================

function calculateVitalsScore(vitals: WebVitals): number {
    let score = 100

    // LCP scoring (good: <2.5s, needs improvement: <4s, poor: >4s)
    if (vitals.lcp !== null) {
        if (vitals.lcp > 4000) score -= 30
        else if (vitals.lcp > 2500) score -= 15
    }

    // FID scoring (good: <100ms, needs improvement: <300ms, poor: >300ms)
    if (vitals.fid !== null) {
        if (vitals.fid > 300) score -= 25
        else if (vitals.fid > 100) score -= 10
    }

    // CLS scoring (good: <0.1, needs improvement: <0.25, poor: >0.25)
    if (vitals.cls !== null) {
        if (vitals.cls > 0.25) score -= 25
        else if (vitals.cls > 0.1) score -= 10
    }

    // TTFB scoring (good: <800ms, needs improvement: <1800ms, poor: >1800ms)
    if (vitals.ttfb !== null) {
        if (vitals.ttfb > 1800) score -= 20
        else if (vitals.ttfb > 800) score -= 10
    }

    return Math.max(0, score)
}

function calculateMemoryScore(memory: MemoryInfo | null): number {
    if (!memory) return 100

    // Score based on heap usage percentage
    if (memory.usagePercent > 90) return 20
    if (memory.usagePercent > 70) return 50
    if (memory.usagePercent > 50) return 75
    return 100
}

function calculateNetworkScore(network: NetworkMetrics): number {
    let score = 100

    // Penalize for failed requests
    const failureRate = network.failedRequests / Math.max(network.totalRequests, 1)
    score -= failureRate * 50

    // Penalize for slow requests
    const slowRate = network.slowRequests / Math.max(network.totalRequests, 1)
    score -= slowRate * 30

    // Penalize for high latency
    if (network.averageLatency > 2000) score -= 20
    else if (network.averageLatency > 1000) score -= 10

    return Math.max(0, score)
}

function calculateOverallScore(breakdown: PerformanceScore['breakdown']): number {
    return Math.round(
        breakdown.vitals * 0.4 +
        breakdown.memory * 0.2 +
        breakdown.network * 0.3 +
        breakdown.rendering * 0.1
    )
}

function getGrade(score: number): PerformanceScore['grade'] {
    if (score >= 90) return 'A'
    if (score >= 75) return 'B'
    if (score >= 60) return 'C'
    if (score >= 40) return 'D'
    return 'F'
}

// =============================================================================
// AI Recommendations Generator
// =============================================================================

function generateRecommendations(metrics: PerformanceMetrics): AIRecommendation[] {
    const recommendations: AIRecommendation[] = []

    // Web Vitals recommendations
    if (metrics.webVitals.lcp && metrics.webVitals.lcp > 2500) {
        recommendations.push({
            id: 'lcp-slow',
            category: metrics.webVitals.lcp > 4000 ? 'critical' : 'warning',
            title: 'Largest Contentful Paint is slow',
            description: `LCP is ${(metrics.webVitals.lcp / 1000).toFixed(1)}s, which is ${metrics.webVitals.lcp > 4000 ? 'poor' : 'needs improvement'}. Users may perceive the page as slow to load.`,
            impact: 'high',
            action: 'Consider lazy loading images, optimizing server response time, or using a CDN.'
        })
    }

    if (metrics.webVitals.cls && metrics.webVitals.cls > 0.1) {
        recommendations.push({
            id: 'cls-high',
            category: metrics.webVitals.cls > 0.25 ? 'critical' : 'warning',
            title: 'Layout Shift detected',
            description: `CLS score is ${metrics.webVitals.cls.toFixed(3)}, indicating visible layout shifts during page load.`,
            impact: 'medium',
            action: 'Set explicit dimensions on images and avoid inserting content above existing content.'
        })
    }

    if (metrics.webVitals.fid && metrics.webVitals.fid > 100) {
        recommendations.push({
            id: 'fid-slow',
            category: metrics.webVitals.fid > 300 ? 'critical' : 'warning',
            title: 'First Input Delay is high',
            description: `FID is ${metrics.webVitals.fid.toFixed(0)}ms. Users may experience lag when interacting with the page.`,
            impact: 'high',
            action: 'Break up long JavaScript tasks and defer non-critical JavaScript.'
        })
    }

    // Memory recommendations
    if (metrics.memory && metrics.memory.usagePercent > 70) {
        recommendations.push({
            id: 'memory-high',
            category: metrics.memory.usagePercent > 90 ? 'critical' : 'warning',
            title: 'High memory usage detected',
            description: `Memory usage is at ${metrics.memory.usagePercent.toFixed(0)}% of available heap.`,
            impact: 'high',
            action: 'Check for memory leaks, clean up event listeners, and optimize data structures.'
        })
    }

    // Network recommendations
    if (metrics.network.averageLatency > 1000) {
        recommendations.push({
            id: 'network-slow',
            category: 'warning',
            title: 'Slow API response times',
            description: `Average API latency is ${(metrics.network.averageLatency / 1000).toFixed(1)}s.`,
            impact: 'medium',
            action: 'Consider implementing request caching, pagination, or optimizing backend queries.'
        })
    }

    if (metrics.network.failedRequests > 0) {
        const failureRate = (metrics.network.failedRequests / Math.max(metrics.network.totalRequests, 1)) * 100
        recommendations.push({
            id: 'network-errors',
            category: failureRate > 10 ? 'critical' : 'warning',
            title: 'Network errors detected',
            description: `${metrics.network.failedRequests} requests failed (${failureRate.toFixed(1)}% failure rate).`,
            impact: 'high',
            action: 'Check network connectivity and verify backend service health.'
        })
    }

    // Long tasks recommendations
    if (metrics.longTasks > 5) {
        recommendations.push({
            id: 'long-tasks',
            category: 'warning',
            title: 'Multiple long tasks detected',
            description: `${metrics.longTasks} tasks took longer than 50ms, potentially blocking the main thread.`,
            impact: 'medium',
            action: 'Use Web Workers for heavy computations or break up long-running tasks.'
        })
    }

    // FPS recommendations
    if (metrics.fps < 30) {
        recommendations.push({
            id: 'low-fps',
            category: 'warning',
            title: 'Low frame rate detected',
            description: `Current FPS is ${metrics.fps}, which may cause janky animations.`,
            impact: 'medium',
            action: 'Reduce DOM complexity, use CSS transforms, and avoid layout thrashing.'
        })
    }

    // Add optimization suggestions if everything is good
    if (recommendations.length === 0) {
        recommendations.push({
            id: 'all-good',
            category: 'info',
            title: 'Performance looks good!',
            description: 'All metrics are within acceptable ranges. Keep up the good work!',
            impact: 'low'
        })
    }

    return recommendations
}

// =============================================================================
// Hook
// =============================================================================

export function usePerformanceMonitor(options: { enabled?: boolean; interval?: number } = {}) {
    const { enabled = true, interval = 5000 } = options

    const [metrics, setMetrics] = useState<PerformanceMetrics>({
        webVitals: { lcp: null, fid: null, cls: null, ttfb: null, fcp: null, inp: null },
        memory: null,
        network: { totalRequests: 0, failedRequests: 0, averageLatency: 0, slowRequests: 0, cachedRequests: 0 },
        fps: 60,
        longTasks: 0,
        renderTime: 0
    })

    const [score, setScore] = useState<PerformanceScore>({
        overall: 100,
        breakdown: { vitals: 100, memory: 100, network: 100, rendering: 100 },
        grade: 'A'
    })

    const [recommendations, setRecommendations] = useState<AIRecommendation[]>([])

    const frameCountRef = useRef(0)
    const lastFrameTimeRef = useRef(performance.now())
    const networkRequestsRef = useRef<{ latency: number; failed: boolean; cached: boolean }[]>([])

    // FPS tracking
    useEffect(() => {
        if (!enabled) return

        let animationFrameId: number

        const measureFps = () => {
            frameCountRef.current++
            const now = performance.now()

            if (now - lastFrameTimeRef.current >= 1000) {
                const fps = Math.round(frameCountRef.current * 1000 / (now - lastFrameTimeRef.current))
                setMetrics(prev => ({ ...prev, fps }))
                frameCountRef.current = 0
                lastFrameTimeRef.current = now
            }

            animationFrameId = requestAnimationFrame(measureFps)
        }

        animationFrameId = requestAnimationFrame(measureFps)

        return () => cancelAnimationFrame(animationFrameId)
    }, [enabled])

    // Long task observer
    useEffect(() => {
        if (!enabled || typeof PerformanceObserver === 'undefined') return

        let longTaskCount = 0

        const observer = new PerformanceObserver((list) => {
            for (const entry of list.getEntries()) {
                if (entry.duration > 50) {
                    longTaskCount++
                }
            }
            setMetrics(prev => ({ ...prev, longTasks: longTaskCount }))
        })

        try {
            observer.observe({ entryTypes: ['longtask'] })
        } catch (e) {
            // longtask not supported
        }

        return () => observer.disconnect()
    }, [enabled])

    // Memory monitoring
    useEffect(() => {
        if (!enabled) return

        const measureMemory = () => {
            // @ts-ignore - memory is non-standard
            if (performance.memory) {
                // @ts-ignore
                const memory = performance.memory
                setMetrics(prev => ({
                    ...prev,
                    memory: {
                        usedJSHeapSize: memory.usedJSHeapSize,
                        totalJSHeapSize: memory.totalJSHeapSize,
                        jsHeapSizeLimit: memory.jsHeapSizeLimit,
                        usagePercent: (memory.usedJSHeapSize / memory.jsHeapSizeLimit) * 100
                    }
                }))
            }
        }

        measureMemory()
        const intervalId = setInterval(measureMemory, interval)

        return () => clearInterval(intervalId)
    }, [enabled, interval])

    // Web Vitals observation
    useEffect(() => {
        if (!enabled || typeof PerformanceObserver === 'undefined') return

        // LCP Observer
        const lcpObserver = new PerformanceObserver((list) => {
            const entries = list.getEntries()
            const lastEntry = entries[entries.length - 1] as PerformanceEntry & { startTime: number }
            setMetrics(prev => ({
                ...prev,
                webVitals: { ...prev.webVitals, lcp: lastEntry.startTime }
            }))
        })

        // FCP Observer
        const fcpObserver = new PerformanceObserver((list) => {
            const entries = list.getEntries()
            const fcpEntry = entries.find(e => e.name === 'first-contentful-paint')
            if (fcpEntry) {
                setMetrics(prev => ({
                    ...prev,
                    webVitals: { ...prev.webVitals, fcp: fcpEntry.startTime }
                }))
            }
        })

        try {
            lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] })
            fcpObserver.observe({ entryTypes: ['paint'] })
        } catch (e) {
            // Observer not supported
        }

        // TTFB from navigation timing
        const navEntry = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming
        if (navEntry) {
            setMetrics(prev => ({
                ...prev,
                webVitals: { ...prev.webVitals, ttfb: navEntry.responseStart - navEntry.requestStart }
            }))
        }

        return () => {
            lcpObserver.disconnect()
            fcpObserver.disconnect()
        }
    }, [enabled])

    // Record network request
    const recordNetworkRequest = useCallback((latency: number, failed: boolean = false, cached: boolean = false) => {
        networkRequestsRef.current.push({ latency, failed, cached })

        // Keep only last 100 requests
        if (networkRequestsRef.current.length > 100) {
            networkRequestsRef.current = networkRequestsRef.current.slice(-100)
        }

        const requests = networkRequestsRef.current
        const totalRequests = requests.length
        const failedRequests = requests.filter(r => r.failed).length
        const cachedRequests = requests.filter(r => r.cached).length
        const slowRequests = requests.filter(r => r.latency > 2000).length
        const averageLatency = requests.reduce((sum, r) => sum + r.latency, 0) / totalRequests

        setMetrics(prev => ({
            ...prev,
            network: { totalRequests, failedRequests, cachedRequests, slowRequests, averageLatency }
        }))
    }, [])

    // Calculate score and recommendations
    useEffect(() => {
        const breakdown = {
            vitals: calculateVitalsScore(metrics.webVitals),
            memory: calculateMemoryScore(metrics.memory),
            network: calculateNetworkScore(metrics.network),
            rendering: metrics.fps >= 60 ? 100 : Math.round((metrics.fps / 60) * 100)
        }

        const overall = calculateOverallScore(breakdown)
        const grade = getGrade(overall)

        setScore({ overall, breakdown, grade })
        setRecommendations(generateRecommendations(metrics))
    }, [metrics])

    return {
        metrics,
        score,
        recommendations,
        recordNetworkRequest,
        isEnabled: enabled
    }
}

export type { PerformanceMetrics, PerformanceScore, AIRecommendation, WebVitals, MemoryInfo, NetworkMetrics }
export default usePerformanceMonitor

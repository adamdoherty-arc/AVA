/**
 * Modern API Client Infrastructure
 * =================================
 *
 * Production-grade API client with:
 * - Type-safe API calls
 * - Automatic retry with exponential backoff
 * - Request/response interceptors
 * - Error transformation
 * - Correlation ID tracking
 * - Request deduplication
 * - Response caching
 *
 * @author AVA Trading Platform
 * @updated 2025-11-29
 */

import axios, {
  AxiosError,
  AxiosInstance,
  AxiosRequestConfig,
  AxiosResponse,
  InternalAxiosRequestConfig,
} from 'axios';

// =============================================================================
// Types
// =============================================================================

export interface APIError {
  code: number;
  name: string;
  message: string;
  details?: Record<string, unknown>;
  timestamp: string;
}

export interface APIResponse<T> {
  success: boolean;
  data?: T;
  message?: string;
  timestamp: string;
}

export interface APIErrorResponse {
  error: APIError;
}

export interface RequestOptions extends Omit<AxiosRequestConfig, 'url' | 'method'> {
  /** Skip response caching */
  skipCache?: boolean;
  /** Custom retry count */
  retries?: number;
  /** Deduplicate concurrent identical requests */
  dedupe?: boolean;
}

export interface APIClientConfig {
  baseURL: string;
  timeout?: number;
  retries?: number;
  retryDelay?: number;
  enableCache?: boolean;
  cacheTTL?: number;
}

// =============================================================================
// Response Cache
// =============================================================================

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  expiresAt: number;
}

class ResponseCache {
  private cache = new Map<string, CacheEntry<unknown>>();
  private defaultTTL: number;

  constructor(ttlMs: number = 30000) {
    this.defaultTTL = ttlMs;
  }

  private generateKey(config: AxiosRequestConfig): string {
    const { method, url, params, data } = config;
    return `${method}:${url}:${JSON.stringify(params)}:${JSON.stringify(data)}`;
  }

  get<T>(config: AxiosRequestConfig): T | null {
    const key = this.generateKey(config);
    const entry = this.cache.get(key);

    if (!entry) return null;

    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return null;
    }

    return entry.data as T;
  }

  set<T>(config: AxiosRequestConfig, data: T, ttl?: number): void {
    const key = this.generateKey(config);
    const now = Date.now();

    this.cache.set(key, {
      data,
      timestamp: now,
      expiresAt: now + (ttl || this.defaultTTL),
    });
  }

  invalidate(pattern?: string): void {
    if (!pattern) {
      this.cache.clear();
      return;
    }

    for (const key of this.cache.keys()) {
      if (key.includes(pattern)) {
        this.cache.delete(key);
      }
    }
  }

  get size(): number {
    return this.cache.size;
  }
}

// =============================================================================
// Request Deduplication
// =============================================================================

class RequestDeduplicator {
  private pending = new Map<string, Promise<AxiosResponse>>();

  private generateKey(config: AxiosRequestConfig): string {
    const { method, url, params, data } = config;
    return `${method}:${url}:${JSON.stringify(params)}:${JSON.stringify(data)}`;
  }

  async dedupe<T>(
    config: AxiosRequestConfig,
    executor: () => Promise<AxiosResponse<T>>
  ): Promise<AxiosResponse<T>> {
    const key = this.generateKey(config);

    // Check if identical request is already in flight
    const existing = this.pending.get(key);
    if (existing) {
      return existing as Promise<AxiosResponse<T>>;
    }

    // Execute and track the request
    const promise = executor().finally(() => {
      this.pending.delete(key);
    });

    this.pending.set(key, promise);
    return promise;
  }
}

// =============================================================================
// API Client Class
// =============================================================================

export class APIClient {
  private client: AxiosInstance;
  private cache: ResponseCache;
  private deduplicator: RequestDeduplicator;
  private config: Required<APIClientConfig>;

  constructor(config: APIClientConfig) {
    this.config = {
      baseURL: config.baseURL,
      timeout: config.timeout ?? 30000,
      retries: config.retries ?? 3,
      retryDelay: config.retryDelay ?? 1000,
      enableCache: config.enableCache ?? true,
      cacheTTL: config.cacheTTL ?? 30000,
    };

    this.cache = new ResponseCache(this.config.cacheTTL);
    this.deduplicator = new RequestDeduplicator();

    this.client = axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor
    this.client.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        // Add correlation ID
        const correlationId = crypto.randomUUID();
        config.headers['X-Correlation-ID'] = correlationId;

        // Add timestamp
        (config as unknown as { metadata: { startTime: number } }).metadata = {
          startTime: Date.now(),
        };

        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response: AxiosResponse) => {
        // Calculate request duration
        const metadata = (response.config as unknown as { metadata?: { startTime: number } }).metadata;
        if (metadata?.startTime) {
          const duration = Date.now() - metadata.startTime;
          console.debug(`[API] ${response.config.method?.toUpperCase()} ${response.config.url} - ${response.status} (${duration}ms)`);
        }
        return response;
      },
      (error: AxiosError<APIErrorResponse>) => {
        // Transform error for consistent handling
        const apiError = this.transformError(error);
        return Promise.reject(apiError);
      }
    );
  }

  private transformError(error: AxiosError<APIErrorResponse>): APIError {
    // Network error
    if (!error.response) {
      return {
        code: 0,
        name: 'NETWORK_ERROR',
        message: error.message || 'Network error occurred',
        timestamp: new Date().toISOString(),
      };
    }

    // API error response
    if (error.response.data?.error) {
      return error.response.data.error;
    }

    // Generic HTTP error
    return {
      code: error.response.status,
      name: `HTTP_${error.response.status}`,
      message: error.response.statusText || 'An error occurred',
      timestamp: new Date().toISOString(),
    };
  }

  private shouldRetry(error: AxiosError, attempt: number, maxRetries: number): boolean {
    if (attempt >= maxRetries) return false;

    // Retry on network errors
    if (!error.response) return true;

    // Retry on 5xx errors and 429 (rate limit)
    const status = error.response.status;
    return status >= 500 || status === 429;
  }

  private async executeWithRetry<T>(
    config: AxiosRequestConfig,
    options: RequestOptions = {}
  ): Promise<AxiosResponse<T>> {
    const maxRetries = options.retries ?? this.config.retries;
    let lastError: AxiosError | null = null;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        const response = await this.client.request<T>(config);
        return response;
      } catch (error) {
        lastError = error as AxiosError;

        if (!this.shouldRetry(lastError, attempt, maxRetries)) {
          throw lastError;
        }

        // Exponential backoff
        const delay = this.config.retryDelay * Math.pow(2, attempt);
        await new Promise((resolve) => setTimeout(resolve, delay));

        console.warn(`[API] Retry ${attempt + 1}/${maxRetries} for ${config.url}`);
      }
    }

    throw lastError;
  }

  // ===========================================================================
  // Public Methods
  // ===========================================================================

  async get<T>(url: string, options: RequestOptions = {}): Promise<T> {
    const config: AxiosRequestConfig = {
      method: 'GET',
      url,
      params: options.params,
      ...options,
    };

    // Check cache for GET requests
    if (this.config.enableCache && !options.skipCache) {
      const cached = this.cache.get<T>(config);
      if (cached) {
        console.debug(`[API] Cache hit for ${url}`);
        return cached;
      }
    }

    // Execute with optional deduplication
    const executor = () => this.executeWithRetry<T>(config, options);
    const response = options.dedupe !== false
      ? await this.deduplicator.dedupe(config, executor)
      : await executor();

    // Cache the response
    if (this.config.enableCache && !options.skipCache) {
      this.cache.set(config, response.data);
    }

    return response.data;
  }

  async post<T, D = unknown>(url: string, data?: D, options: RequestOptions = {}): Promise<T> {
    const config: AxiosRequestConfig = {
      method: 'POST',
      url,
      data,
      ...options,
    };

    const response = await this.executeWithRetry<T>(config, options);

    // Invalidate relevant cache entries on mutations
    this.cache.invalidate(url.split('/')[0]);

    return response.data;
  }

  async put<T, D = unknown>(url: string, data?: D, options: RequestOptions = {}): Promise<T> {
    const config: AxiosRequestConfig = {
      method: 'PUT',
      url,
      data,
      ...options,
    };

    const response = await this.executeWithRetry<T>(config, options);
    this.cache.invalidate(url.split('/')[0]);
    return response.data;
  }

  async patch<T, D = unknown>(url: string, data?: D, options: RequestOptions = {}): Promise<T> {
    const config: AxiosRequestConfig = {
      method: 'PATCH',
      url,
      data,
      ...options,
    };

    const response = await this.executeWithRetry<T>(config, options);
    this.cache.invalidate(url.split('/')[0]);
    return response.data;
  }

  async delete<T>(url: string, options: RequestOptions = {}): Promise<T> {
    const config: AxiosRequestConfig = {
      method: 'DELETE',
      url,
      ...options,
    };

    const response = await this.executeWithRetry<T>(config, options);
    this.cache.invalidate(url.split('/')[0]);
    return response.data;
  }

  /**
   * Stream response using Server-Sent Events
   */
  stream(url: string, onMessage: (data: string) => void, options: RequestOptions = {}): () => void {
    const fullUrl = `${this.config.baseURL}${url}`;
    const params = new URLSearchParams(options.params as Record<string, string>).toString();
    const eventSource = new EventSource(params ? `${fullUrl}?${params}` : fullUrl);

    eventSource.onmessage = (event) => {
      onMessage(event.data);
    };

    eventSource.onerror = (error) => {
      console.error('[API] SSE error:', error);
    };

    // Return cleanup function
    return () => {
      eventSource.close();
    };
  }

  /**
   * Invalidate cache entries
   */
  invalidateCache(pattern?: string): void {
    this.cache.invalidate(pattern);
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): { size: number } {
    return { size: this.cache.size };
  }
}

// =============================================================================
// Default Client Instance
// =============================================================================

export const apiClient = new APIClient({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8002/api',
  timeout: 30000,
  retries: 3,
  enableCache: true,
  cacheTTL: 30000,
});

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Type guard for API errors
 */
export function isAPIError(error: unknown): error is APIError {
  return (
    typeof error === 'object' &&
    error !== null &&
    'code' in error &&
    'name' in error &&
    'message' in error
  );
}

/**
 * Format API error for display
 */
export function formatAPIError(error: unknown): string {
  if (isAPIError(error)) {
    return error.message;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return 'An unexpected error occurred';
}

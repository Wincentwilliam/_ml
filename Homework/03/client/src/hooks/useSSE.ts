import { useState, useEffect, useCallback, useRef } from 'react';

export interface TrainingUpdate {
  epoch: number;
  loss: number;
  gradient_norms: Record<string, number>;
  learning_rate: number;
  optimizer: string;
  is_complete: boolean;
  error?: string;
}

export interface TrainingStatus {
  is_training: boolean;
  epoch: number;
  total_epochs: number;
  current_loss: number;
  learning_rate: number;
  optimizer: string;
}

interface UseSSEOptions {
  url?: string;
  autoConnect?: boolean;
  onMessage?: (data: TrainingUpdate) => void;
}

/**
 * Hook for connecting to Server-Sent Events training stream.
 * Handles reconnection, buffering, and state management.
 */
export function useSSE(options: UseSSEOptions = {}) {
  const {
    url = '/train/stream',
    autoConnect = false,
    onMessage
  } = options;

  const [connected, setConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<TrainingUpdate | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [updateCount, setUpdateCount] = useState(0);

  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  const disconnect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    if (reconnectTimeoutRef.current !== null) {
      window.clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    setConnected(false);
  }, []);

  const connect = useCallback(() => {
    // Close existing connection
    disconnect();

    try {
      const es = new EventSource(url);
      eventSourceRef.current = es;

      es.onopen = () => {
        setConnected(true);
        setError(null);
        console.log('[SSE] Connected to training stream');
      };

      es.onmessage = (event) => {
        try {
          const data: TrainingUpdate = JSON.parse(event.data);
          setLastUpdate(data);
          setUpdateCount(prev => prev + 1);

          if (onMessage) {
            onMessage(data);
          }

          // Auto-disconnect on completion
          if (data.is_complete || data.error) {
            setTimeout(() => disconnect(), 1000);
          }
        } catch (e) {
          console.error('[SSE] Parse error:', e);
        }
      };

      es.onerror = (e) => {
        console.error('[SSE] Connection error:', e);
        setConnected(false);

        // Attempt reconnection
        if (!reconnectTimeoutRef.current) {
          reconnectTimeoutRef.current = window.setTimeout(() => {
            connect();
          }, 3000);
        }
      };
    } catch (e) {
      setError(`Failed to connect: ${e}`);
      setConnected(false);
    }
  }, [url, disconnect, onMessage]);

  useEffect(() => {
    if (autoConnect) {
      connect();
    }
    return () => disconnect();
  }, [autoConnect, connect, disconnect]);

  return {
    connected,
    lastUpdate,
    error,
    updateCount,
    connect,
    disconnect
  };
}

/**
 * Hook for training control operations.
 */
export function useTrainingControl(apiBaseUrl = '') {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${apiBaseUrl}/train/status`);
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
        return data;
      }
    } catch (e) {
      console.error('Failed to fetch status:', e);
    }
    return null;
  }, [apiBaseUrl]);

  // Poll status every 2 seconds to keep UI updated
  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const startTraining = useCallback(async (params: {
    learning_rate: number;
    optimizer: 'sgd' | 'adam' | 'rmsprop';
    epochs: number;
    update_interval: number;
  }) => {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${apiBaseUrl}/train/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });

      if (res.ok) {
        const data = await res.json();
        await fetchStatus();
        return { success: true, data };
      } else {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to start training');
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
      return { success: false, error: e };
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl, fetchStatus]);

  const stopTraining = useCallback(async () => {
    try {
      const res = await fetch(`${apiBaseUrl}/train/stop`, { method: 'POST' });
      if (res.ok) {
        await fetchStatus();
        return { success: true };
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to stop training');
    }
    return { success: false };
  }, [apiBaseUrl, fetchStatus]);

  const resetTraining = useCallback(async () => {
    try {
      const res = await fetch(`${apiBaseUrl}/train/reset`, { method: 'POST' });
      if (res.ok) {
        setStatus(null);
        return { success: true };
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to reset training');
    }
    return { success: false };
  }, [apiBaseUrl]);

  return {
    status,
    loading,
    error,
    startTraining,
    stopTraining,
    resetTraining,
    refreshStatus: fetchStatus
  };
}

/**
 * Hook for fetching model state.
 */
export function useModelState(apiBaseUrl = '') {
  const [modelState, setModelState] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchModelState = useCallback(async () => {
    setLoading(true);
    try {
      const res = await fetch(`${apiBaseUrl}/train/model`);
      if (res.ok) {
        const data = await res.json();
        setModelState(data);
      }
    } catch (e) {
      console.error('Failed to fetch model state:', e);
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl]);

  return { modelState, loading, fetchModelState };
}
import { useCallback, useEffect, useState } from 'react';
import { errorMessage } from './api';

interface Settled<T> {
  fetcher: (() => Promise<T>) | null;
  nonce: number;
  data: T | null;
  error: string;
}

/**
 * Runs `fetcher` whenever its identity changes (wrap it in useCallback) or
 * reload() is called. While a new request is in flight, `data` keeps the
 * previous result and `loading` is true, so the UI can dim instead of blanking.
 */
export function useFetch<T>(fetcher: (() => Promise<T>) | null) {
  const [nonce, setNonce] = useState(0);
  const [settled, setSettled] = useState<Settled<T>>({ fetcher: null, nonce: 0, data: null, error: '' });

  useEffect(() => {
    if (!fetcher) return;
    let cancelled = false;
    fetcher()
      .then(data => { if (!cancelled) setSettled({ fetcher, nonce, data, error: '' }); })
      .catch(e => { if (!cancelled) setSettled(prev => ({ fetcher, nonce, data: prev.data, error: errorMessage(e) })); });
    return () => { cancelled = true; };
  }, [fetcher, nonce]);

  const current = settled.fetcher === fetcher && settled.nonce === nonce;
  const reload = useCallback(() => setNonce(n => n + 1), []);
  return {
    data: settled.data,
    error: current ? settled.error : '',
    loading: fetcher !== null && !current,
    reload,
  };
}

import { useEffect, useRef, useState } from 'react';
import { Activity, AlertTriangle, Send, Sparkles } from 'lucide-react';
import { errorMessage, sendChat } from '../api';
import type { ChatStatus, ChatTurn } from '../api';

interface Message {
  from: 'user' | 'bot' | 'error';
  text: string;
  source?: 'gemini' | 'claude' | 'rules';
  notice?: string | null;
}

const SUGGESTIONS: Record<string, string[]> = {
  doctor:  ['Give me an overall summary', 'Any risk factors?', 'Why this forecast?', 'How is their sleep?'],
  patient: ['How am I doing?', 'How is my sleep?', 'What should I change this week?', 'Explain my forecast'],
};

interface Props {
  patientId: number | null;
  status: ChatStatus | null;
}

export default function ChatPanel({ patientId, status }: Props) {
  const [role, setRole]         = useState<'doctor' | 'patient'>('doctor');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput]       = useState('');
  const [sending, setSending]   = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  // A different patient or audience is a different conversation
  useEffect(() => { setMessages([]); }, [patientId, role]);

  useEffect(() => { endRef.current?.scrollIntoView({ block: 'end' }); }, [messages, sending]);

  const send = async (text: string) => {
    const msg = text.trim();
    if (!msg || sending) return;
    const history: ChatTurn[] = messages
      .filter(m => m.from !== 'error')
      .map(m => ({ role: m.from === 'user' ? 'user' : 'assistant', content: m.text }));
    setMessages(prev => [...prev, { from: 'user', text: msg }]);
    setInput('');
    setSending(true);
    try {
      const res = await sendChat(msg, role, patientId, history);
      setMessages(prev => [...prev, { from: 'bot', text: res.reply, source: res.source, notice: res.notice }]);
    } catch (e) {
      setMessages(prev => [...prev, { from: 'error', text: errorMessage(e) }]);
    } finally {
      setSending(false);
    }
  };

  const llm = status?.llm_enabled;

  return (
    <aside className="chat-panel">
      <div className="brand">
        <span className="brand-mark"><Activity size={18} /></span>
        <h2>HealthAI</h2>
      </div>

      <div className="chat-controls">
        <div className="segmented" role="group" aria-label="Who is asking">
          {(['doctor', 'patient'] as const).map(r => (
            <button key={r} className={role === r ? 'active' : ''} aria-pressed={role === r}
              onClick={() => setRole(r)}>
              {r === 'doctor' ? 'Doctor' : 'Patient'}
            </button>
          ))}
        </div>
        {status && (
          <div className={`assistant-mode ${llm ? 'on' : 'off'}`}>
            <Sparkles size={13} />
            {llm
              ? `Answers by ${status.provider === 'gemini' ? 'Gemini' : 'Claude'}`
              : 'Built-in answers — add an API key in server/.env for AI answers'}
          </div>
        )}
      </div>

      <div className="chat-window" aria-live="polite">
        {messages.length === 0 && (
          <div className="chat-empty">
            <p>
              {role === 'doctor'
                ? `Ask about patient ${patientId ?? ''}'s activity, sleep, heart rate or forecast.`
                : 'Ask how you are doing, or about your sleep, steps or forecast.'}
            </p>
            <div className="chips">
              {SUGGESTIONS[role].map(s => (
                <button key={s} className="chip" onClick={() => send(s)}>{s}</button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`chat-row ${m.from}`}>
            <div className={`chat-bubble ${m.from}`}>
              {m.from === 'error' && <AlertTriangle size={14} />}
              {m.text}
            </div>
            {m.notice && <div className="chat-notice">{m.notice}</div>}
          </div>
        ))}
        {sending && (
          <div className="chat-row bot">
            <div className="chat-bubble bot typing" aria-label="Assistant is typing">
              <span /><span /><span />
            </div>
          </div>
        )}
        <div ref={endRef} />
      </div>

      <form className="chat-input-area" onSubmit={e => { e.preventDefault(); send(input); }}>
        <input type="text" placeholder="Ask the health assistant…" aria-label="Message"
          value={input} onChange={e => setInput(e.target.value)} />
        <button type="submit" disabled={sending || !input.trim()} aria-label="Send">
          <Send size={16} />
        </button>
      </form>
    </aside>
  );
}

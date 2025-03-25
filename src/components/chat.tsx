import { Send } from 'lucide-react';
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { useAuthStore } from '@/lib/store';

const FOLDERS = [
  'ITG hypotheses',
  "Tom's analysis",
  'Accenture AB tests',
  'Everything combined',
] as const;

type Message = {
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
};

const QUICK_LINKS = [
  {
    title: 'Mobile Content Engagement on US Range Rover Nameplate Pages',
    href: '#mobile-content',
  },
  {
    title: 'Nameplate Visualiser Model Selection',
    href: '#nameplate-visualiser',
  },
  {
    title: 'US Forms Copy Optimization',
    href: '#forms-copy',
  },
  {
    title: 'Hybrid Vehicle Search Trends in the US',
    href: '#hybrid-trends',
  },
];

export function Chat() {
  const navigate = useNavigate();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [selectedFolder, setSelectedFolder] = useState<typeof FOLDERS[number]>('ITG hypotheses');
  const [isLoading, setIsLoading] = useState(false);
  const { logout } = useAuthStore();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMessage }),
      });

      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.answer, sources: data.sources },
      ]);
    } catch (error) {
      console.error('Error:', error);
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: 'Sorry, there was an error processing your request.' },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  return (
    <div className="flex h-screen flex-col">
      {/* Notices */}
      <div className="space-y-1 text-center text-sm">
        <div className="bg-gray-100 px-4 py-2 text-gray-700">
          Data collection notice: All conversations are stored for analysis purposes
        </div>
        <div className="bg-gray-50 px-4 py-2 text-gray-600">
          Note: This knowledge assistant can make mistakes. Please verify important information.
        </div>
      </div>

      {/* Header with folder selection and logout */}
      <div className="border-b px-4 py-2 flex justify-between items-center">
        <select
          value={selectedFolder}
          onChange={(e) => setSelectedFolder(e.target.value as typeof FOLDERS[number])}
          className="rounded-md border-gray-300 py-2 pl-3 pr-10 text-sm focus:border-[#e7298a] focus:outline-none focus:ring-[#e7298a]"
        >
          {FOLDERS.map((folder) => (
            <option key={folder} value={folder}>
              {folder}
            </option>
          ))}
        </select>
        <Button variant="outline" onClick={handleLogout} size="sm">
          Logout
        </Button>
      </div>

      {/* Chat messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 max-w-3xl mx-auto w-full">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full space-y-8">
            <h1 className="text-2xl font-semibold text-gray-700">How can I help you?</h1>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-2xl">
              {QUICK_LINKS.map((link) => (
                <a
                  key={link.href}
                  href={link.href}
                  className="p-4 border rounded-lg hover:bg-gray-50 transition-colors duration-200 text-gray-700 hover:text-gray-900"
                >
                  {link.title}
                </a>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${
                message.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              <div
                className={`max-w-[80%] rounded-lg px-4 py-2 ${
                  message.role === 'user'
                    ? 'bg-[#e7298a] text-white'
                    : 'bg-gray-100 text-gray-900'
                }`}
              >
                <p>{message.content}</p>
                {message.sources && (
                  <div className="mt-2 text-sm opacity-75">
                    <p>Sources:</p>
                    <ul className="list-inside list-disc">
                      {message.sources.map((source, i) => (
                        <li key={i}>{source}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="flex justify-start">
            <div className="max-w-[80%] rounded-lg bg-gray-100 px-4 py-2">
              <div className="flex space-x-2">
                <div className="h-2 w-2 animate-bounce rounded-full bg-gray-400"></div>
                <div className="h-2 w-2 animate-bounce rounded-full bg-gray-400" style={{ animationDelay: '0.2s' }}></div>
                <div className="h-2 w-2 animate-bounce rounded-full bg-gray-400" style={{ animationDelay: '0.4s' }}></div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input form */}
      <form onSubmit={handleSubmit} className="border-t p-4 max-w-3xl mx-auto w-full">
        <div className="flex space-x-4">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            disabled={isLoading}
            className="flex-1"
          />
          <Button type="submit" disabled={isLoading}>
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </form>
    </div>
  );
}
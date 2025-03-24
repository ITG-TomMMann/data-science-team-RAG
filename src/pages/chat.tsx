import { ChatInterface } from '../components/chat_interface';

export default function Chat() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-white to-pink-50">
      <div className="chat-container">
        <ChatInterface />
      </div>
    </div>
  );
}

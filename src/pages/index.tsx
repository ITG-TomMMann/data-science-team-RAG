import { AuthScreen } from '../components/auth_screen';
import { useRouter } from 'next/router';
import '../styles/globals.css';

export default function Home() {
  const router = useRouter();

  const handleLogin = async (email: string, password: string) => {
    // Add your authentication logic here
    // For now, just redirect to chat
    router.push('/chat');
  };

  return <AuthScreen onLogin={handleLogin} />;
}

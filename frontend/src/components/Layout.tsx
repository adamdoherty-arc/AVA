import { Outlet } from 'react-router-dom'
import { Sidebar } from './Sidebar'
import { AvaChatWidget } from './AvaChatWidget'

export function Layout() {
    return (
        <div className="flex h-screen bg-background text-text overflow-hidden">
            <Sidebar />
            <main className="flex-1 overflow-auto flex justify-center">
                <div className="min-h-full p-6 lg:p-8 animate-fade-in w-full" style={{ maxWidth: '1400px' }}>
                    <Outlet />
                </div>
            </main>
            {/* Floating AVA Chat Widget - available on all pages */}
            <AvaChatWidget />
        </div>
    )
}

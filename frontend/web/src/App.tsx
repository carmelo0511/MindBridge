import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { Helmet } from 'react-helmet-async'
import { Box } from '@mui/material'
import { motion } from 'framer-motion'

import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Analytics from './pages/Analytics'
import Users from './pages/Users'
import Interventions from './pages/Interventions'
import Privacy from './pages/Privacy'
import Settings from './pages/Settings'
import Login from './pages/Login'
import NotFound from './pages/NotFound'

// Mock auth state - in real app, this would come from context/store
const useAuth = () => {
  const [isAuthenticated, setIsAuthenticated] = React.useState(true) // Mock authenticated state
  return { isAuthenticated, setIsAuthenticated }
}

function App() {
  const { isAuthenticated } = useAuth()

  if (!isAuthenticated) {
    return <Login />
  }

  return (
    <>
      <Helmet>
        <title>MindBridge - Mental Health Analytics Dashboard</title>
        <meta
          name="description"
          content="Professional dashboard for mental health analytics, privacy-preserving insights, and intervention tracking."
        />
      </Helmet>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
      >
        <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
          <Layout>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/analytics" element={<Analytics />} />
              <Route path="/users" element={<Users />} />
              <Route path="/interventions" element={<Interventions />} />
              <Route path="/privacy" element={<Privacy />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </Layout>
        </Box>
      </motion.div>
    </>
  )
}

export default App
import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { Helmet } from 'react-helmet-async'
import { Box } from '@mui/material'
import { motion } from 'framer-motion'

import PersonalDashboard from './pages/PersonalDashboard'
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
        <title>MindBridge - Votre Compagnon de Bien-être Mental</title>
        <meta
          name="description"
          content="Votre espace privé pour le suivi de votre bien-être mental. 100% privé, 100% local, pour votre sérénité."
        />
      </Helmet>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.3 }}
      >
        <Box sx={{ minHeight: '100vh' }}>
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<PersonalDashboard />} />
            <Route path="/interventions" element={<Interventions />} />
            <Route path="/privacy" element={<Privacy />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </Box>
      </motion.div>
    </>
  )
}

export default App
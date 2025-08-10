import React from 'react'
import { Box, Typography, Button } from '@mui/material'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'

const NotFound: React.FC = () => {
  const navigate = useNavigate()

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '60vh',
          textAlign: 'center',
        }}
      >
        <Typography variant="h1" fontWeight={700} color="primary" sx={{ mb: 2 }}>
          404
        </Typography>
        <Typography variant="h4" fontWeight={600} gutterBottom>
          Page introuvable
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
          La page que vous recherchez n'existe pas ou a été déplacée.
        </Typography>
        <Button
          variant="contained"
          onClick={() => navigate('/dashboard')}
          size="large"
        >
          Retour au Tableau de Bord
        </Button>
      </Box>
    </motion.div>
  )
}

export default NotFound
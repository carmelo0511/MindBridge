import React from 'react'
import { Box, Typography, Card, CardContent } from '@mui/material'
import { motion } from 'framer-motion'

const Settings: React.FC = () => {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <Box>
        <Typography variant="h4" fontWeight={700} gutterBottom>
          Paramètres Système
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
          Configuration avancée du système MindBridge
        </Typography>
        
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Page en développement
            </Typography>
            <Typography variant="body1">
              L'interface de paramètres sera disponible prochainement.
            </Typography>
          </CardContent>
        </Card>
      </Box>
    </motion.div>
  )
}

export default Settings
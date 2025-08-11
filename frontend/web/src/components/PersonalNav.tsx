import React from 'react'
import { 
  AppBar, 
  Toolbar, 
  Typography, 
  IconButton, 
  Box,
  Chip,
  Avatar
} from '@mui/material'
import {
  Psychology,
  Security,
  Settings,
  Logout,
  Shield
} from '@mui/icons-material'
import { motion } from 'framer-motion'

const PersonalNav: React.FC = () => {
  return (
    <AppBar 
      position="static" 
      elevation={0}
      sx={{ 
        background: 'transparent',
        backdropFilter: 'blur(20px)',
        borderBottom: '1px solid rgba(255,255,255,0.1)'
      }}
    >
      <Toolbar sx={{ justifyContent: 'space-between', py: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <motion.div
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center' }}>
              <Psychology sx={{ color: 'white', fontSize: 32, mr: 1 }} />
              <Typography 
                variant="h5" 
                component="div" 
                sx={{ 
                  fontWeight: 700,
                  background: 'linear-gradient(45deg, #fff, rgba(255,255,255,0.8))',
                  backgroundClip: 'text',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent'
                }}
              >
                MindBridge
              </Typography>
            </Box>
          </motion.div>
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Chip
            icon={<Shield sx={{ fontSize: 16 }} />}
            label="100% Privé"
            variant="outlined"
            size="small"
            sx={{ 
              color: 'white',
              borderColor: 'rgba(255,255,255,0.3)',
              '& .MuiChip-icon': { color: '#4CAF50' }
            }}
          />
          
          <IconButton sx={{ color: 'rgba(255,255,255,0.8)' }}>
            <Settings />
          </IconButton>
          
          <Avatar 
            sx={{ 
              width: 32, 
              height: 32,
              background: 'linear-gradient(45deg, rgba(255,255,255,0.2), rgba(255,255,255,0.1))',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(255,255,255,0.3)'
            }}
          >
            😊
          </Avatar>
        </Box>
      </Toolbar>
    </AppBar>
  )
}

export default PersonalNav
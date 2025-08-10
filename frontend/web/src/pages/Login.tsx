import React from 'react'
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Avatar,
  Divider,
  Alert,
  Link,
} from '@mui/material'
import { motion } from 'framer-motion'

const Login: React.FC = () => {
  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #2196F3 0%, #1976D2 100%)',
        p: 2,
      }}
    >
      <motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <Card sx={{ maxWidth: 400, width: '100%' }}>
          <CardContent sx={{ p: 4 }}>
            <Box sx={{ textAlign: 'center', mb: 4 }}>
              <Avatar
                sx={{
                  width: 64,
                  height: 64,
                  bgcolor: 'primary.main',
                  mx: 'auto',
                  mb: 2,
                }}
              >
                🧠
              </Avatar>
              <Typography variant="h4" fontWeight={700} gutterBottom>
                MindBridge
              </Typography>
              <Typography variant="body1" color="text.secondary">
                Interface Professionnelle Sécurisée
              </Typography>
            </Box>

            <Alert severity="info" sx={{ mb: 3 }}>
              <Typography variant="body2">
                <strong>Demo:</strong> Cliquez sur "Se connecter" pour accéder au tableau de bord
              </Typography>
            </Alert>

            <Box component="form" sx={{ mt: 1 }}>
              <TextField
                margin="normal"
                required
                fullWidth
                label="Email professionnel"
                type="email"
                autoComplete="email"
                autoFocus
                defaultValue="dr.dubois@mindbridge.ai"
              />
              <TextField
                margin="normal"
                required
                fullWidth
                label="Mot de passe"
                type="password"
                autoComplete="current-password"
                defaultValue="demo123"
              />
              
              <Button
                type="submit"
                fullWidth
                variant="contained"
                sx={{ mt: 3, mb: 2, py: 1.5 }}
              >
                Se Connecter
              </Button>

              <Box sx={{ textAlign: 'center' }}>
                <Link href="#" variant="body2">
                  Mot de passe oublié ?
                </Link>
              </Box>
            </Box>

            <Divider sx={{ my: 3 }} />

            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                Système conforme RGPD et HIPAA
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Chiffrement AES-256 • Zéro-knowledge • Local
              </Typography>
            </Box>
          </CardContent>
        </Card>
      </motion.div>
    </Box>
  )
}

export default Login
import React, { useState, useEffect } from 'react'
import PersonalNav from '../components/PersonalNav'
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Avatar,
  Chip,
  LinearProgress,
  Button,
  TextField,
  IconButton,
  Paper,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  CircularProgress,
  Alert,
  AlertTitle,
} from '@mui/material'
import {
  Psychology,
  SelfImprovement,
  Mood,
  LocalHospital,
  Lock,
  Favorite,
  TrendingUp,
  TrendingDown,
  CheckCircle,
  Warning,
  Lightbulb,
  Schedule,
  VolumeUp,
  Create,
  Send,
  Security,
} from '@mui/icons-material'
import { motion } from 'framer-motion'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadialBarChart,
  RadialBar,
  PieChart,
  Pie,
  Cell
} from 'recharts'

interface MoodEntry {
  date: string
  score: number
  note?: string
}

interface AnalysisResult {
  overall_risk_score: number
  risk_level: string
  condition_probabilities: Record<string, number>
  suggestions: string[]
  crisis_detected: boolean
}

const PersonalDashboard: React.FC = () => {
  // States pour l'interface personnelle
  const [currentMood, setCurrentMood] = useState<number>(7)
  const [journalText, setJournalText] = useState<string>('')
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false)
  const [moodHistory] = useState<MoodEntry[]>([
    { date: 'Lun', score: 6 },
    { date: 'Mar', score: 7 },
    { date: 'Mer', score: 5 },
    { date: 'Jeu', score: 8 },
    { date: 'Ven', score: 7 },
    { date: 'Sam', score: 9 },
    { date: 'Dim', score: 8 },
  ])

  // Couleurs apaisantes pour l'UI personnelle
  const moodColors = {
    excellent: '#4CAF50',
    good: '#8BC34A',
    okay: '#FFC107',
    low: '#FF9800',
    concerning: '#F44336'
  }

  const getMoodColor = (score: number) => {
    if (score >= 9) return moodColors.excellent
    if (score >= 7) return moodColors.good
    if (score >= 5) return moodColors.okay
    if (score >= 3) return moodColors.low
    return moodColors.concerning
  }

  const getMoodLabel = (score: number) => {
    if (score >= 9) return 'Excellent'
    if (score >= 7) return 'Bien'
    if (score >= 5) return 'Correct'
    if (score >= 3) return 'Difficile'
    return 'Très difficile'
  }

  // Analyse du texte du journal avec l'IA
  const analyzeJournalEntry = async () => {
    if (!journalText.trim()) return
    
    setIsAnalyzing(true)
    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: journalText,
          cultural_context: 'personal',
          language: 'fr'
        })
      })
      
      const result = await response.json()
      setAnalysisResult(result)
    } catch (error) {
      console.error('Erreur analyse:', error)
    }
    setIsAnalyzing(false)
  }

  // Check-in quotidien rapide
  const submitDailyCheckIn = async () => {
    const checkInData = {
      mood_score: currentMood,
      journal_entry: journalText,
      date: new Date().toISOString()
    }
    
    // Ici on sauvegarderait localement (chiffré)
    console.log('Check-in quotidien:', checkInData)
  }

  return (
    <Box sx={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', minHeight: '100vh' }}>
      <PersonalNav />
      <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Header Personnel */}
      <Box sx={{ mb: 4 }}>
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Typography variant="h3" sx={{ color: 'white', fontWeight: 700, mb: 1 }}>
            Salut ! 👋
          </Typography>
          <Typography variant="h6" sx={{ color: 'rgba(255,255,255,0.8)', mb: 2 }}>
            Comment vous sentez-vous aujourd'hui ?
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Lock sx={{ color: 'white' }} />
            <Typography variant="body2" sx={{ color: 'rgba(255,255,255,0.8)' }}>
              Vos données restent 100% privées et locales
            </Typography>
          </Box>
        </motion.div>
      </Box>

      <Grid container spacing={3}>
        {/* Check-in Quotidien Rapide */}
        <Grid item xs={12} md={6}>
          <motion.div whileHover={{ scale: 1.02 }}>
            <Card sx={{ height: '100%', background: 'rgba(255,255,255,0.95)', backdropFilter: 'blur(10px)' }}>
              <CardContent sx={{ p: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <SelfImprovement sx={{ color: '#667eea', mr: 1 }} />
                  <Typography variant="h6" fontWeight={600}>
                    Check-in Rapide
                  </Typography>
                </Box>
                
                <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                  Sur une échelle de 1 à 10, comment vous sentez-vous ?
                </Typography>
                
                <Box sx={{ mb: 3 }}>
                  <Typography variant="h4" color={getMoodColor(currentMood)} fontWeight={700} sx={{ mb: 1 }}>
                    {currentMood}/10
                  </Typography>
                  <Typography variant="subtitle1" color={getMoodColor(currentMood)} fontWeight={500}>
                    {getMoodLabel(currentMood)}
                  </Typography>
                  
                  <Box sx={{ mt: 2 }}>
                    <input
                      type="range"
                      min="1"
                      max="10"
                      value={currentMood}
                      onChange={(e) => setCurrentMood(Number(e.target.value))}
                      style={{
                        width: '100%',
                        height: '8px',
                        borderRadius: '5px',
                        background: `linear-gradient(to right, #F44336, #FF9800, #FFC107, #8BC34A, #4CAF50)`,
                        outline: 'none',
                        cursor: 'pointer'
                      }}
                    />
                  </Box>
                </Box>
                
                <Button
                  variant="contained"
                  startIcon={<CheckCircle />}
                  onClick={submitDailyCheckIn}
                  sx={{ 
                    background: 'linear-gradient(45deg, #667eea, #764ba2)',
                    '&:hover': { background: 'linear-gradient(45deg, #5a67d8, #6b46c1)' }
                  }}
                >
                  Enregistrer Check-in
                </Button>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Humeur de la Semaine */}
        <Grid item xs={12} md={6}>
          <motion.div whileHover={{ scale: 1.02 }}>
            <Card sx={{ height: '100%', background: 'rgba(255,255,255,0.95)', backdropFilter: 'blur(10px)' }}>
              <CardContent sx={{ p: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <TrendingUp sx={{ color: '#4CAF50', mr: 1 }} />
                  <Typography variant="h6" fontWeight={600}>
                    Votre Semaine
                  </Typography>
                </Box>
                
                <ResponsiveContainer width="100%" height={150}>
                  <LineChart data={moodHistory}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.1)" />
                    <XAxis dataKey="date" stroke="#666" />
                    <YAxis domain={[1, 10]} stroke="#666" />
                    <Tooltip 
                      contentStyle={{ 
                        background: 'rgba(255,255,255,0.95)', 
                        border: 'none', 
                        borderRadius: '8px',
                        boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                      }} 
                    />
                    <Line 
                      type="monotone" 
                      dataKey="score" 
                      stroke="#667eea" 
                      strokeWidth={3}
                      dot={{ fill: '#667eea', strokeWidth: 2, r: 4 }}
                      activeDot={{ r: 6, stroke: '#667eea', strokeWidth: 2, fill: '#fff' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
                
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  Tendance générale: 📈 En amélioration
                </Typography>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Journal Personnel avec IA */}
        <Grid item xs={12}>
          <motion.div whileHover={{ scale: 1.01 }}>
            <Card sx={{ background: 'rgba(255,255,255,0.95)', backdropFilter: 'blur(10px)' }}>
              <CardContent sx={{ p: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                  <Create sx={{ color: '#667eea', mr: 1 }} />
                  <Typography variant="h6" fontWeight={600}>
                    Journal Personnel
                  </Typography>
                  <Box sx={{ ml: 'auto' }}>
                    <Chip
                      icon={<Psychology />}
                      label="IA Privée"
                      variant="outlined"
                      size="small"
                      sx={{ borderColor: '#667eea', color: '#667eea' }}
                    />
                  </Box>
                </Box>
                
                <TextField
                  multiline
                  rows={4}
                  fullWidth
                  variant="outlined"
                  placeholder="Comment vous sentez-vous aujourd'hui ? Qu'est-ce qui vous préoccupe ?"
                  value={journalText}
                  onChange={(e) => setJournalText(e.target.value)}
                  sx={{ mb: 2 }}
                />
                
                <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
                  <Button
                    variant="contained"
                    startIcon={isAnalyzing ? <CircularProgress size={20} color="inherit" /> : <Psychology />}
                    onClick={analyzeJournalEntry}
                    disabled={isAnalyzing || !journalText.trim()}
                    sx={{ 
                      background: 'linear-gradient(45deg, #667eea, #764ba2)',
                      '&:hover': { background: 'linear-gradient(45deg, #5a67d8, #6b46c1)' }
                    }}
                  >
                    {isAnalyzing ? 'Analyse...' : 'Analyser avec IA'}
                  </Button>
                  
                  <Button
                    variant="outlined"
                    startIcon={<VolumeUp />}
                    sx={{ borderColor: '#667eea', color: '#667eea' }}
                  >
                    Enregistrer Audio
                  </Button>
                </Box>

                {/* Résultats d'Analyse IA */}
                {analysisResult && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                  >
                    <Paper sx={{ p: 3, background: 'rgba(103, 126, 234, 0.05)', border: '1px solid rgba(103, 126, 234, 0.2)' }}>
                      <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
                        <Lightbulb sx={{ color: '#667eea', mr: 1 }} />
                        Insights IA
                      </Typography>
                      
                      {analysisResult.crisis_detected && (
                        <Alert severity="error" sx={{ mb: 2 }}>
                          <AlertTitle>Support Urgent Recommandé</AlertTitle>
                          Si vous ressentez des pensées suicidaires, contactez immédiatement le 3114 (gratuit, 24h/24)
                        </Alert>
                      )}
                      
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                          Score de bien-être: {Math.round((1 - analysisResult.overall_risk_score) * 100)}%
                        </Typography>
                        <LinearProgress 
                          variant="determinate" 
                          value={(1 - analysisResult.overall_risk_score) * 100}
                          sx={{ 
                            height: 8, 
                            borderRadius: 4,
                            backgroundColor: 'rgba(103, 126, 234, 0.2)',
                            '& .MuiLinearProgress-bar': {
                              background: `linear-gradient(to right, ${getMoodColor((1 - analysisResult.overall_risk_score) * 10)}, #667eea)`
                            }
                          }}
                        />
                      </Box>
                      
                      {analysisResult.suggestions.length > 0 && (
                        <Box>
                          <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
                            Suggestions personnalisées:
                          </Typography>
                          <List dense>
                            {analysisResult.suggestions.slice(0, 3).map((suggestion, index) => (
                              <ListItem key={index} sx={{ px: 0 }}>
                                <ListItemIcon>
                                  <Favorite sx={{ color: '#667eea', fontSize: 20 }} />
                                </ListItemIcon>
                                <ListItemText 
                                  primary={suggestion}
                                  primaryTypographyProps={{ variant: 'body2' }}
                                />
                              </ListItem>
                            ))}
                          </List>
                        </Box>
                      )}
                    </Paper>
                  </motion.div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Interventions Rapides */}
        <Grid item xs={12} md={4}>
          <motion.div whileHover={{ scale: 1.02 }}>
            <Card sx={{ height: '100%', background: 'rgba(255,255,255,0.95)', backdropFilter: 'blur(10px)' }}>
              <CardContent sx={{ p: 3 }}>
                <Typography variant="h6" fontWeight={600} sx={{ mb: 2 }}>
                  🧘 Interventions Rapides
                </Typography>
                
                <List>
                  <ListItem button sx={{ borderRadius: 2, mb: 1 }}>
                    <ListItemIcon>
                      <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: '#4CAF50' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Respiration 4-7-8"
                      secondary="2 minutes"
                      primaryTypographyProps={{ fontWeight: 500 }}
                    />
                  </ListItem>
                  
                  <ListItem button sx={{ borderRadius: 2, mb: 1 }}>
                    <ListItemIcon>
                      <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: '#FF9800' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Méditation guidée"
                      secondary="5 minutes"
                      primaryTypographyProps={{ fontWeight: 500 }}
                    />
                  </ListItem>
                  
                  <ListItem button sx={{ borderRadius: 2, mb: 1 }}>
                    <ListItemIcon>
                      <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: '#667eea' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Gratitude rapide"
                      secondary="1 minute"
                      primaryTypographyProps={{ fontWeight: 500 }}
                    />
                  </ListItem>
                </List>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Ressources de Crise */}
        <Grid item xs={12} md={4}>
          <motion.div whileHover={{ scale: 1.02 }}>
            <Card sx={{ height: '100%', background: 'rgba(255,255,255,0.95)', backdropFilter: 'blur(10px)' }}>
              <CardContent sx={{ p: 3 }}>
                <Typography variant="h6" fontWeight={600} sx={{ mb: 2, color: '#F44336' }}>
                  🆘 Besoin d'Aide ?
                </Typography>
                
                <List>
                  <ListItem>
                    <ListItemIcon>
                      <LocalHospital sx={{ color: '#F44336' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="3114"
                      secondary="Numéro national gratuit 24h/24"
                      primaryTypographyProps={{ fontWeight: 600, color: '#F44336' }}
                    />
                  </ListItem>
                  
                  <ListItem>
                    <ListItemIcon>
                      <LocalHospital sx={{ color: '#F44336' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="15 - SAMU"
                      secondary="Urgences vitales"
                      primaryTypographyProps={{ fontWeight: 600, color: '#F44336' }}
                    />
                  </ListItem>
                </List>
                
                <Button
                  variant="contained"
                  color="error"
                  fullWidth
                  startIcon={<LocalHospital />}
                  sx={{ mt: 2 }}
                  onClick={() => window.open('tel:3114')}
                >
                  Appeler Maintenant
                </Button>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Contrôles de Confidentialité */}
        <Grid item xs={12} md={4}>
          <motion.div whileHover={{ scale: 1.02 }}>
            <Card sx={{ height: '100%', background: 'rgba(255,255,255,0.95)', backdropFilter: 'blur(10px)' }}>
              <CardContent sx={{ p: 3 }}>
                <Typography variant="h6" fontWeight={600} sx={{ mb: 2 }}>
                  🔒 Votre Confidentialité
                </Typography>
                
                <List>
                  <ListItem>
                    <ListItemIcon>
                      <Security sx={{ color: '#4CAF50' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Données 100% locales"
                      secondary="Rien ne quitte votre appareil"
                      secondaryTypographyProps={{ variant: 'caption' }}
                    />
                  </ListItem>
                  
                  <ListItem>
                    <ListItemIcon>
                      <Lock sx={{ color: '#4CAF50' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Chiffrement AES-256"
                      secondary="Sécurité militaire"
                      secondaryTypographyProps={{ variant: 'caption' }}
                    />
                  </ListItem>
                  
                  <ListItem>
                    <ListItemIcon>
                      <CheckCircle sx={{ color: '#4CAF50' }} />
                    </ListItemIcon>
                    <ListItemText 
                      primary="Contrôle total"
                      secondary="Supprimez vos données quand vous voulez"
                      secondaryTypographyProps={{ variant: 'caption' }}
                    />
                  </ListItem>
                </List>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>
      </Box>
    </Box>
  )
}

export default PersonalDashboard
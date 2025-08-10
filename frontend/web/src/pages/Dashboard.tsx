import React from 'react'
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Avatar,
  Chip,
  LinearProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Button,
  Alert,
  Paper,
} from '@mui/material'
import {
  TrendingUp,
  TrendingDown,
  Warning,
  Security,
  Psychology,
  People,
  CheckCircle,
  Info,
  Emergency,
  Analytics,
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
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
} from 'recharts'

// Mock data for charts
const weeklyTrendData = [
  { day: 'Lun', riskScore: 0.25, users: 45 },
  { day: 'Mar', riskScore: 0.32, users: 52 },
  { day: 'Mer', riskScore: 0.28, users: 48 },
  { day: 'Jeu', riskScore: 0.35, users: 61 },
  { day: 'Ven', riskScore: 0.29, users: 43 },
  { day: 'Sam', riskScore: 0.22, users: 38 },
  { day: 'Dim', riskScore: 0.26, users: 41 },
]

const conditionDistribution = [
  { name: 'Anxiété', value: 35, color: '#FF9800' },
  { name: 'Dépression', value: 28, color: '#3F51B5' },
  { name: 'Burnout', value: 20, color: '#FF5722' },
  { name: 'PTSD', value: 12, color: '#795548' },
  { name: 'Bipolaire', value: 5, color: '#9C27B0' },
]

const riskLevelData = [
  { level: 'Faible', count: 234, color: '#4CAF50' },
  { level: 'Modéré', count: 89, color: '#FF9800' },
  { level: 'Élevé', count: 23, color: '#F44336' },
  { level: 'Critique', count: 3, color: '#D32F2F' },
]

interface StatCardProps {
  title: string
  value: string | number
  subtitle?: string
  trend?: 'up' | 'down' | 'neutral'
  trendValue?: string
  icon: React.ReactNode
  color: string
}

const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  subtitle,
  trend,
  trendValue,
  icon,
  color,
}) => (
  <motion.div
    whileHover={{ scale: 1.02 }}
    transition={{ type: 'spring', stiffness: 300 }}
  >
    <Card sx={{ height: '100%', position: 'relative', overflow: 'hidden' }}>
      <Box
        sx={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: 4,
          bgcolor: color,
        }}
      />
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
          <Box sx={{ flex: 1 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              {title}
            </Typography>
            <Typography variant="h4" fontWeight={700}>
              {value}
            </Typography>
            {subtitle && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                {subtitle}
              </Typography>
            )}
            {trend && trendValue && (
              <Box sx={{ display: 'flex', alignItems: 'center', mt: 1, gap: 0.5 }}>
                {trend === 'up' ? (
                  <TrendingUp sx={{ fontSize: 16, color: 'success.main' }} />
                ) : trend === 'down' ? (
                  <TrendingDown sx={{ fontSize: 16, color: 'error.main' }} />
                ) : null}
                <Typography
                  variant="body2"
                  color={trend === 'up' ? 'success.main' : trend === 'down' ? 'error.main' : 'text.secondary'}
                  fontWeight={500}
                >
                  {trendValue}
                </Typography>
              </Box>
            )}
          </Box>
          <Avatar
            sx={{
              bgcolor: `${color}15`,
              color: color,
              width: 48,
              height: 48,
            }}
          >
            {icon}
          </Avatar>
        </Box>
      </CardContent>
    </Card>
  </motion.div>
)

const Dashboard: React.FC = () => {
  return (
    <Box>
      <motion.div
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <Box sx={{ mb: 4 }}>
          <Typography variant="h4" fontWeight={700} gutterBottom>
            Tableau de Bord
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Vue d'ensemble des métriques de santé mentale et de la confidentialité
          </Typography>
        </Box>
      </motion.div>

      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Utilisateurs Actifs"
            value="2,847"
            subtitle="Dernières 24h"
            trend="up"
            trendValue="+12.5%"
            icon={<People />}
            color="#2196F3"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Score de Risque Moyen"
            value="0.28"
            subtitle="Population générale"
            trend="down"
            trendValue="-3.2%"
            icon={<Psychology />}
            color="#4CAF50"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Interventions Actives"
            value="156"
            subtitle="En cours"
            trend="up"
            trendValue="+8 nouvelles"
            icon={<CheckCircle />}
            color="#FF9800"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            title="Niveau de Confidentialité"
            value="Maximum"
            subtitle="98.7% conformité"
            icon={<Security />}
            color="#1B5E20"
          />
        </Grid>
      </Grid>

      {/* Alerts and Notifications */}
      <motion.div
        initial={{ x: -20, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ delay: 0.2 }}
      >
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid item xs={12} md={8}>
            <Alert
              severity="warning"
              icon={<Warning />}
              action={
                <Button color="inherit" size="small">
                  Voir Détails
                </Button>
              }
              sx={{ mb: 2 }}
            >
              <Typography variant="subtitle2" gutterBottom>
                3 utilisateurs nécessitent une attention immédiate
              </Typography>
              <Typography variant="body2">
                Niveaux de risque élevés détectés dans les dernières 2 heures
              </Typography>
            </Alert>

            <Alert
              severity="info"
              icon={<Info />}
              action={
                <Button color="inherit" size="small">
                  Configurer
                </Button>
              }
            >
              <Typography variant="subtitle2" gutterBottom>
                Mise à jour du modèle IA disponible
              </Typography>
              <Typography variant="body2">
                Nouvelles améliorations pour la détection culturelle
              </Typography>
            </Alert>
          </Grid>

          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Actions Rapides
                </Typography>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  <Button
                    variant="contained"
                    startIcon={<Emergency />}
                    color="error"
                    fullWidth
                  >
                    Mode Crise
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<Analytics />}
                    fullWidth
                  >
                    Rapport Hebdo
                  </Button>
                  <Button
                    variant="outlined"
                    startIcon={<Security />}
                    fullWidth
                  >
                    Audit Confidentialité
                  </Button>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </motion.div>

      {/* Charts and Analytics */}
      <Grid container spacing={3}>
        {/* Weekly Trend */}
        <Grid item xs={12} lg={8}>
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                  <Typography variant="h6">
                    Tendance Hebdomadaire
                  </Typography>
                  <Chip
                    label="7 derniers jours"
                    size="small"
                    variant="outlined"
                  />
                </Box>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={weeklyTrendData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="day" />
                    <YAxis />
                    <Tooltip
                      labelFormatter={(label) => `Jour: ${label}`}
                      formatter={(value, name) => [
                        name === 'riskScore' ? value.toFixed(2) : value,
                        name === 'riskScore' ? 'Score de Risque' : 'Utilisateurs'
                      ]}
                    />
                    <Line
                      type="monotone"
                      dataKey="riskScore"
                      stroke="#F44336"
                      strokeWidth={2}
                      dot={{ fill: '#F44336' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Risk Distribution */}
        <Grid item xs={12} lg={4}>
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.4 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Distribution des Risques
                </Typography>
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie
                      data={riskLevelData}
                      cx="50%"
                      cy="50%"
                      innerRadius={40}
                      outerRadius={80}
                      dataKey="count"
                    >
                      {riskLevelData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
                <List dense>
                  {riskLevelData.map((item, index) => (
                    <ListItem key={index} sx={{ py: 0.5 }}>
                      <ListItemIcon sx={{ minWidth: 24 }}>
                        <Box
                          sx={{
                            width: 12,
                            height: 12,
                            borderRadius: '50%',
                            bgcolor: item.color,
                          }}
                        />
                      </ListItemIcon>
                      <ListItemText
                        primary={item.level}
                        secondary={`${item.count} utilisateurs`}
                      />
                    </ListItem>
                  ))}
                </List>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Condition Distribution */}
        <Grid item xs={12} lg={6}>
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Conditions Détectées
                </Typography>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={conditionDistribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="value" fill="#2196F3" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Recent Activity */}
        <Grid item xs={12} lg={6}>
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.6 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Activité Récente
                </Typography>
                <List>
                  {[
                    { text: 'Nouvelle intervention déployée pour anxiété', time: 'Il y a 5 min', severity: 'success' },
                    { text: 'Utilisateur #2847 nécessite attention', time: 'Il y a 12 min', severity: 'warning' },
                    { text: 'Rapport de confidentialité généré', time: 'Il y a 1h', severity: 'info' },
                    { text: 'Mise à jour du modèle IA complétée', time: 'Il y a 2h', severity: 'success' },
                  ].map((item, index) => (
                    <React.Fragment key={index}>
                      <ListItem>
                        <ListItemIcon>
                          <Box
                            sx={{
                              width: 8,
                              height: 8,
                              borderRadius: '50%',
                              bgcolor: item.severity === 'success' ? 'success.main' :
                                      item.severity === 'warning' ? 'warning.main' : 'info.main',
                            }}
                          />
                        </ListItemIcon>
                        <ListItemText
                          primary={item.text}
                          secondary={item.time}
                        />
                      </ListItem>
                      {index < 3 && <Divider />}
                    </React.Fragment>
                  ))}
                </List>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>
    </Box>
  )
}

export default Dashboard
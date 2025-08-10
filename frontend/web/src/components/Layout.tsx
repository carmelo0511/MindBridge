import React, { useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import {
  Box,
  Drawer,
  AppBar,
  Toolbar,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  IconButton,
  Typography,
  Avatar,
  Menu,
  MenuItem,
  Divider,
  Badge,
  useTheme,
  useMediaQuery,
} from '@mui/material'
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Analytics as AnalyticsIcon,
  People as PeopleIcon,
  Psychology as PsychologyIcon,
  Security as SecurityIcon,
  Settings as SettingsIcon,
  Notifications as NotificationsIcon,
  AccountCircle,
  Logout,
  Help,
  Brightness4,
  Brightness7,
} from '@mui/icons-material'
import { motion } from 'framer-motion'

const drawerWidth = 280

interface LayoutProps {
  children: React.ReactNode
}

interface NavItem {
  id: string
  label: string
  icon: React.ComponentType
  path: string
  badge?: number
}

const navItems: NavItem[] = [
  {
    id: 'dashboard',
    label: 'Tableau de Bord',
    icon: DashboardIcon,
    path: '/dashboard',
  },
  {
    id: 'analytics',
    label: 'Analytiques',
    icon: AnalyticsIcon,
    path: '/analytics',
  },
  {
    id: 'users',
    label: 'Utilisateurs',
    icon: PeopleIcon,
    path: '/users',
    badge: 3, // Mock notifications
  },
  {
    id: 'interventions',
    label: 'Interventions',
    icon: PsychologyIcon,
    path: '/interventions',
  },
  {
    id: 'privacy',
    label: 'Confidentialité',
    icon: SecurityIcon,
    path: '/privacy',
  },
  {
    id: 'settings',
    label: 'Paramètres',
    icon: SettingsIcon,
    path: '/settings',
  },
]

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down('lg'))
  const location = useLocation()
  const navigate = useNavigate()
  
  const [mobileOpen, setMobileOpen] = useState(false)
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null)
  const [darkMode, setDarkMode] = useState(false)

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen)
  }

  const handleProfileMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget)
  }

  const handleProfileMenuClose = () => {
    setAnchorEl(null)
  }

  const handleThemeToggle = () => {
    setDarkMode(!darkMode)
    // In a real app, this would update the theme context
  }

  const drawer = (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Logo and Brand */}
      <Box sx={{ p: 3, borderBottom: 1, borderColor: 'divider' }}>
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.3 }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Avatar
              sx={{
                width: 40,
                height: 40,
                background: 'linear-gradient(135deg, #2196F3 0%, #1976D2 100%)',
              }}
            >
              🧠
            </Avatar>
            <Box>
              <Typography variant="h6" fontWeight={700} className="text-gradient">
                MindBridge
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Tableau Professionnel
              </Typography>
            </Box>
          </Box>
        </motion.div>
      </Box>

      {/* Navigation */}
      <Box sx={{ flex: 1, py: 2 }}>
        <List>
          {navItems.map((item, index) => {
            const isActive = location.pathname === item.path
            const Icon = item.icon

            return (
              <motion.div
                key={item.id}
                initial={{ x: -20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: index * 0.1 }}
              >
                <ListItem disablePadding sx={{ px: 2, mb: 0.5 }}>
                  <ListItemButton
                    onClick={() => {
                      navigate(item.path)
                      if (isMobile) setMobileOpen(false)
                    }}
                    sx={{
                      borderRadius: 2,
                      bgcolor: isActive ? 'primary.main' : 'transparent',
                      color: isActive ? 'white' : 'text.primary',
                      '&:hover': {
                        bgcolor: isActive ? 'primary.dark' : 'action.hover',
                      },
                      transition: 'all 0.2s ease-in-out',
                    }}
                  >
                    <ListItemIcon
                      sx={{
                        color: 'inherit',
                        minWidth: 36,
                      }}
                    >
                      <Badge badgeContent={item.badge} color="error">
                        <Icon />
                      </Badge>
                    </ListItemIcon>
                    <ListItemText
                      primary={item.label}
                      primaryTypographyProps={{
                        fontWeight: isActive ? 600 : 400,
                      }}
                    />
                  </ListItemButton>
                </ListItem>
              </motion.div>
            )
          })}
        </List>
      </Box>

      {/* User Info */}
      <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider' }}>
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 2,
            p: 1.5,
            borderRadius: 2,
            bgcolor: 'background.paper',
            border: 1,
            borderColor: 'divider',
          }}
        >
          <Avatar sx={{ width: 32, height: 32 }}>
            👨‍⚕️
          </Avatar>
          <Box sx={{ flex: 1, minWidth: 0 }}>
            <Typography variant="body2" fontWeight={600} noWrap>
              Dr. Marie Dubois
            </Typography>
            <Typography variant="caption" color="text.secondary" noWrap>
              Psychologue Clinicien
            </Typography>
          </Box>
        </Box>
      </Box>
    </Box>
  )

  return (
    <>
      {/* App Bar */}
      <AppBar
        position="fixed"
        elevation={0}
        sx={{
          width: { lg: `calc(100% - ${drawerWidth}px)` },
          ml: { lg: `${drawerWidth}px` },
          bgcolor: 'background.paper',
          color: 'text.primary',
          borderBottom: 1,
          borderColor: 'divider',
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { lg: 'none' } }}
          >
            <MenuIcon />
          </IconButton>

          <Box sx={{ flex: 1 }}>
            <Typography variant="h6" noWrap fontWeight={600}>
              {navItems.find(item => item.path === location.pathname)?.label || 'MindBridge'}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Interface professionnelle sécurisée
            </Typography>
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <IconButton onClick={handleThemeToggle}>
              {darkMode ? <Brightness7 /> : <Brightness4 />}
            </IconButton>

            <IconButton>
              <Badge badgeContent={7} color="error">
                <NotificationsIcon />
              </Badge>
            </IconButton>

            <IconButton
              onClick={handleProfileMenuOpen}
              sx={{ ml: 1 }}
            >
              <AccountCircle />
            </IconButton>
          </Box>
        </Toolbar>
      </AppBar>

      {/* Profile Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleProfileMenuClose}
        PaperProps={{
          sx: { minWidth: 200 }
        }}
      >
        <MenuItem onClick={handleProfileMenuClose}>
          <ListItemIcon>
            <AccountCircle fontSize="small" />
          </ListItemIcon>
          Mon Profil
        </MenuItem>
        <MenuItem onClick={handleProfileMenuClose}>
          <ListItemIcon>
            <Help fontSize="small" />
          </ListItemIcon>
          Aide
        </MenuItem>
        <Divider />
        <MenuItem onClick={handleProfileMenuClose}>
          <ListItemIcon>
            <Logout fontSize="small" />
          </ListItemIcon>
          Se Déconnecter
        </MenuItem>
      </Menu>

      {/* Navigation Drawer */}
      <Box
        component="nav"
        sx={{ width: { lg: drawerWidth }, flexShrink: { lg: 0 } }}
      >
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true,
          }}
          sx={{
            display: { xs: 'block', lg: 'none' },
            '& .MuiDrawer-paper': { 
              boxSizing: 'border-box', 
              width: drawerWidth,
              bgcolor: 'background.paper',
            },
          }}
        >
          {drawer}
        </Drawer>
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', lg: 'block' },
            '& .MuiDrawer-paper': { 
              boxSizing: 'border-box', 
              width: drawerWidth,
              bgcolor: 'background.paper',
              borderRight: 1,
              borderColor: 'divider',
            },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { lg: `calc(100% - ${drawerWidth}px)` },
          mt: { xs: 7, sm: 8 },
        }}
      >
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.4 }}
        >
          {children}
        </motion.div>
      </Box>
    </>
  )
}

export default Layout
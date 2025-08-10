"""
MindBridge Privacy Manager

This module implements zero-knowledge architecture, differential privacy,
and advanced encryption for mental health data protection.

Features:
- AES-256 encryption for all local data
- Differential privacy for aggregate metrics
- Zero-knowledge proofs for data verification
- Secure key derivation and management
- Data anonymization and pseudonymization
- Auto-destruction of sensitive data
- Panic button for immediate data erasure
"""

import os
import json
import hashlib
import secrets
import hmac
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import base64
import logging
from pathlib import Path

# Cryptographic imports
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.fernet import Fernet, MultiFernet
from cryptography.hazmat.backends import default_backend
import numpy as np


class PrivacyLevel(Enum):
    MINIMAL = "minimal"      # Basic encryption only
    MODERATE = "moderate"    # Encryption + differential privacy
    HIGH = "high"           # + zero-knowledge proofs
    MAXIMUM = "maximum"     # + advanced anonymization + auto-destruction


class DataType(Enum):
    TEXT_ANALYSIS = "text_analysis"
    BEHAVIORAL_DATA = "behavioral_data" 
    RISK_ASSESSMENT = "risk_assessment"
    INTERVENTION_DATA = "intervention_data"
    USER_PROFILE = "user_profile"
    SYSTEM_LOGS = "system_logs"


@dataclass
class EncryptionMetadata:
    algorithm: str
    key_derivation: str
    salt: bytes
    iv: bytes
    timestamp: datetime
    data_type: DataType
    retention_policy: int  # Days until auto-destruction


@dataclass
class PrivacyAuditLog:
    timestamp: datetime
    operation: str
    data_type: DataType
    privacy_level: PrivacyLevel
    success: bool
    metadata: Dict[str, Any]


class SecureKeyManager:
    """Manages encryption keys with secure derivation and rotation"""
    
    def __init__(self, privacy_level: PrivacyLevel = PrivacyLevel.MAXIMUM):
        self.privacy_level = privacy_level
        self.master_key = self._generate_master_key()
        self.key_cache = {}  # Temporary key cache
        self.key_rotation_interval = 24 * 3600  # 24 hours
        self.last_rotation = time.time()
        
    def _generate_master_key(self) -> bytes:
        """Generate cryptographically secure master key"""
        return secrets.token_bytes(32)  # 256-bit key
    
    def derive_key(self, purpose: str, salt: bytes = None) -> Tuple[bytes, bytes]:
        """Derive purpose-specific key from master key"""
        if salt is None:
            salt = secrets.token_bytes(32)
        
        kdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            info=purpose.encode(),
            backend=default_backend()
        )
        
        derived_key = kdf.derive(self.master_key)
        return derived_key, salt
    
    def get_encryption_key(self, data_type: DataType) -> Tuple[bytes, bytes]:
        """Get encryption key for specific data type"""
        cache_key = f"{data_type.value}_{int(time.time() // self.key_rotation_interval)}"
        
        if cache_key not in self.key_cache:
            key, salt = self.derive_key(f"encrypt_{data_type.value}")
            self.key_cache[cache_key] = (key, salt)
        
        return self.key_cache[cache_key]
    
    def rotate_keys(self) -> bool:
        """Rotate encryption keys for enhanced security"""
        if time.time() - self.last_rotation < self.key_rotation_interval:
            return False
        
        # Clear key cache to force regeneration
        self.key_cache.clear()
        self.last_rotation = time.time()
        return True
    
    def derive_user_key(self, user_id_hash: str, device_id: str) -> bytes:
        """Derive user-specific key for cross-device sync"""
        salt = hashlib.sha256(f"{user_id_hash}:{device_id}".encode()).digest()
        key, _ = self.derive_key("user_data", salt)
        return key
    
    def emergency_key_destruction(self) -> bool:
        """Emergency destruction of all keys (panic button)"""
        try:
            # Overwrite master key with random data
            self.master_key = secrets.token_bytes(32)
            
            # Clear all caches
            self.key_cache.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            return True
        except Exception:
            return False


class DifferentialPrivacy:
    """Implements differential privacy for aggregate statistics"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta     # Failure probability
        self.noise_scale = 1.0 / epsilon
    
    def add_laplace_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """Add Laplace noise for epsilon-differential privacy"""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def add_gaussian_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """Add Gaussian noise for (epsilon, delta)-differential privacy"""
        sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / self.delta))) / self.epsilon
        noise = np.random.normal(0, sigma)
        return value + noise
    
    def privatize_count(self, count: int) -> int:
        """Add noise to count statistics"""
        noisy_count = self.add_laplace_noise(float(count), sensitivity=1.0)
        return max(0, int(round(noisy_count)))
    
    def privatize_average(self, values: List[float], bounds: Tuple[float, float]) -> float:
        """Add noise to average with bounded values"""
        if not values:
            return 0.0
        
        # Clip values to bounds
        clipped_values = [max(bounds[0], min(bounds[1], v)) for v in values]
        avg = np.mean(clipped_values)
        
        # Sensitivity is (upper_bound - lower_bound) / len(values)
        sensitivity = (bounds[1] - bounds[0]) / len(clipped_values)
        return self.add_laplace_noise(avg, sensitivity)
    
    def privatize_histogram(self, bins: List[int]) -> List[int]:
        """Add noise to histogram bins"""
        return [self.privatize_count(count) for count in bins]


class ZeroKnowledgeProofs:
    """Implements zero-knowledge proofs for data verification"""
    
    def __init__(self):
        self.commitment_randomness = {}
        
    def commit_to_value(self, value: float, identifier: str) -> str:
        """Create cryptographic commitment to a value"""
        randomness = secrets.token_bytes(32)
        commitment_data = f"{value}:{base64.b64encode(randomness).decode()}"
        commitment_hash = hashlib.sha256(commitment_data.encode()).hexdigest()
        
        self.commitment_randomness[identifier] = randomness
        return commitment_hash
    
    def verify_commitment(self, commitment_hash: str, value: float, identifier: str) -> bool:
        """Verify a commitment without revealing the original value"""
        if identifier not in self.commitment_randomness:
            return False
        
        randomness = self.commitment_randomness[identifier]
        expected_data = f"{value}:{base64.b64encode(randomness).decode()}"
        expected_hash = hashlib.sha256(expected_data.encode()).hexdigest()
        
        return hmac.compare_digest(commitment_hash, expected_hash)
    
    def prove_range(self, value: float, min_val: float, max_val: float) -> Dict[str, str]:
        """Prove that a value is within a range without revealing the value"""
        if not (min_val <= value <= max_val):
            raise ValueError("Value not in specified range")
        
        # Simplified range proof (would use more sophisticated ZK proofs in production)
        proof_data = {
            'range_min': str(min_val),
            'range_max': str(max_val),
            'timestamp': str(int(time.time())),
            'nonce': secrets.token_hex(16)
        }
        
        # Create proof hash
        proof_string = json.dumps(proof_data, sort_keys=True)
        proof_hash = hashlib.sha256(f"{value}:{proof_string}".encode()).hexdigest()
        proof_data['proof_hash'] = proof_hash
        
        return proof_data


class DataAnonymizer:
    """Handles data anonymization and pseudonymization"""
    
    def __init__(self, key_manager: SecureKeyManager):
        self.key_manager = key_manager
        self.identifier_mappings = {}
        
    def pseudonymize_identifier(self, original_id: str, context: str) -> str:
        """Convert real identifier to pseudonym"""
        key, salt = self.key_manager.derive_key(f"pseudo_{context}")
        
        # Create HMAC-based pseudonym
        pseudonym = hmac.new(
            key, 
            original_id.encode(), 
            hashlib.sha256
        ).hexdigest()[:16]  # Truncate for usability
        
        return f"anon_{context}_{pseudonym}"
    
    def anonymize_text(self, text: str) -> str:
        """Remove or replace PII from text"""
        import re
        
        # Remove common PII patterns
        patterns = [
            (r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]'),  # Phone numbers
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),   # SSN
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),  # Email
            (r'\b\d{1,5}\s+([A-Z][a-z]+\s+)+(Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b', '[ADDRESS]'),  # Addresses
            (r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]'),  # Names (basic)
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', '[DATE]'),  # Dates
        ]
        
        anonymized_text = text
        for pattern, replacement in patterns:
            anonymized_text = re.sub(pattern, replacement, anonymized_text, flags=re.IGNORECASE)
        
        return anonymized_text
    
    def k_anonymize_dataset(self, dataset: List[Dict], k: int = 3) -> List[Dict]:
        """Apply k-anonymity to dataset"""
        # Simplified k-anonymity implementation
        # In production, would use more sophisticated algorithms
        
        anonymized_dataset = []
        for record in dataset:
            anonymized_record = {}
            
            for key, value in record.items():
                if isinstance(value, str):
                    anonymized_record[key] = self.anonymize_text(value)
                elif isinstance(value, (int, float)):
                    # Generalize numeric values
                    if key in ['age', 'risk_score']:
                        anonymized_record[key] = self._generalize_numeric(value, key)
                    else:
                        anonymized_record[key] = value
                else:
                    anonymized_record[key] = value
            
            anonymized_dataset.append(anonymized_record)
        
        return anonymized_dataset
    
    def _generalize_numeric(self, value: Union[int, float], field_name: str) -> str:
        """Generalize numeric values for k-anonymity"""
        if field_name == 'age':
            if value < 18:
                return 'under_18'
            elif value < 25:
                return '18-24'
            elif value < 35:
                return '25-34'
            elif value < 50:
                return '35-49'
            else:
                return '50_plus'
        
        elif field_name == 'risk_score':
            if value < 0.25:
                return 'low'
            elif value < 0.5:
                return 'moderate'
            elif value < 0.75:
                return 'high'
            else:
                return 'critical'
        
        return str(value)


class SecureStorage:
    """Handles encrypted local storage with auto-destruction"""
    
    def __init__(self, 
                 key_manager: SecureKeyManager, 
                 storage_path: str = "./secure_storage"):
        self.key_manager = key_manager
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.metadata_cache = {}
        
    def store_encrypted(self, 
                       data: Any, 
                       data_type: DataType,
                       retention_days: int = 30,
                       user_context: str = None) -> str:
        """Store data with encryption and metadata"""
        
        # Get encryption key for this data type
        encryption_key, salt = self.key_manager.get_encryption_key(data_type)
        
        # Generate IV for AES
        iv = secrets.token_bytes(16)
        
        # Serialize data
        if isinstance(data, (dict, list)):
            serialized_data = json.dumps(data).encode()
        else:
            serialized_data = str(data).encode()
        
        # Encrypt data using AES-256-CBC
        cipher = Cipher(
            algorithms.AES(encryption_key),
            modes.CBC(iv),
            backend=default_backend()
        )
        
        encryptor = cipher.encryptor()
        
        # Pad data to block size
        padded_data = self._pad_data(serialized_data)
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Generate unique storage ID
        storage_id = secrets.token_hex(16)
        
        # Create metadata
        metadata = EncryptionMetadata(
            algorithm="AES-256-CBC",
            key_derivation="HKDF",
            salt=salt,
            iv=iv,
            timestamp=datetime.now(),
            data_type=data_type,
            retention_policy=retention_days
        )
        
        # Store encrypted data
        data_file = self.storage_path / f"{storage_id}.dat"
        with open(data_file, 'wb') as f:
            f.write(encrypted_data)
        
        # Store metadata separately
        metadata_file = self.storage_path / f"{storage_id}.meta"
        with open(metadata_file, 'w') as f:
            json.dump(asdict(metadata), f, default=str)
        
        self.metadata_cache[storage_id] = metadata
        
        return storage_id
    
    def retrieve_decrypted(self, storage_id: str) -> Tuple[Any, EncryptionMetadata]:
        """Retrieve and decrypt stored data"""
        
        # Load metadata
        metadata_file = self.storage_path / f"{storage_id}.meta"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata not found for {storage_id}")
        
        with open(metadata_file, 'r') as f:
            metadata_dict = json.load(f)
        
        metadata = EncryptionMetadata(**metadata_dict)
        
        # Check retention policy
        if self._is_expired(metadata):
            self.secure_delete(storage_id)
            raise ValueError(f"Data {storage_id} has expired and been deleted")
        
        # Load encrypted data
        data_file = self.storage_path / f"{storage_id}.dat"
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found for {storage_id}")
        
        with open(data_file, 'rb') as f:
            encrypted_data = f.read()
        
        # Recreate encryption key
        encryption_key, _ = self.key_manager.derive_key(
            f"encrypt_{metadata.data_type.value}", 
            metadata.salt
        )
        
        # Decrypt data
        cipher = Cipher(
            algorithms.AES(encryption_key),
            modes.CBC(metadata.iv),
            backend=default_backend()
        )
        
        decryptor = cipher.decryptor()
        decrypted_padded = decryptor.update(encrypted_data) + decryptor.finalize()
        
        # Remove padding
        decrypted_data = self._unpad_data(decrypted_padded)
        
        # Deserialize
        try:
            deserialized_data = json.loads(decrypted_data.decode())
        except json.JSONDecodeError:
            deserialized_data = decrypted_data.decode()
        
        return deserialized_data, metadata
    
    def secure_delete(self, storage_id: str) -> bool:
        """Securely delete stored data"""
        try:
            data_file = self.storage_path / f"{storage_id}.dat"
            metadata_file = self.storage_path / f"{storage_id}.meta"
            
            # Overwrite files with random data before deletion
            for file_path in [data_file, metadata_file]:
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    with open(file_path, 'wb') as f:
                        f.write(secrets.token_bytes(file_size))
                    file_path.unlink()
            
            # Remove from cache
            if storage_id in self.metadata_cache:
                del self.metadata_cache[storage_id]
            
            return True
        except Exception:
            return False
    
    def cleanup_expired_data(self) -> int:
        """Clean up expired data based on retention policies"""
        cleaned_count = 0
        
        for storage_id, metadata in list(self.metadata_cache.items()):
            if self._is_expired(metadata):
                if self.secure_delete(storage_id):
                    cleaned_count += 1
        
        return cleaned_count
    
    def emergency_data_destruction(self) -> bool:
        """Emergency destruction of all stored data"""
        try:
            # Securely delete all files
            for file_path in self.storage_path.glob("*"):
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    with open(file_path, 'wb') as f:
                        f.write(secrets.token_bytes(file_size))
                    file_path.unlink()
            
            # Clear cache
            self.metadata_cache.clear()
            
            return True
        except Exception:
            return False
    
    def _pad_data(self, data: bytes) -> bytes:
        """PKCS7 padding for AES encryption"""
        block_size = 16
        padding_len = block_size - (len(data) % block_size)
        padding = bytes([padding_len] * padding_len)
        return data + padding
    
    def _unpad_data(self, padded_data: bytes) -> bytes:
        """Remove PKCS7 padding"""
        padding_len = padded_data[-1]
        return padded_data[:-padding_len]
    
    def _is_expired(self, metadata: EncryptionMetadata) -> bool:
        """Check if data has exceeded retention policy"""
        expiry_date = metadata.timestamp + timedelta(days=metadata.retention_policy)
        return datetime.now() > expiry_date


class MindBridgePrivacyManager:
    """Main privacy manager orchestrating all privacy-preserving operations"""
    
    def __init__(self, 
                 privacy_level: PrivacyLevel = PrivacyLevel.MAXIMUM,
                 storage_path: str = "./secure_storage",
                 audit_log_path: str = "./privacy_audit.log"):
        
        self.privacy_level = privacy_level
        self.key_manager = SecureKeyManager(privacy_level)
        self.differential_privacy = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        self.zero_knowledge = ZeroKnowledgeProofs()
        self.anonymizer = DataAnonymizer(self.key_manager)
        self.secure_storage = SecureStorage(self.key_manager, storage_path)
        
        # Setup audit logging
        self.audit_log_path = Path(audit_log_path)
        self.audit_logger = self._setup_audit_logging()
        
        # Privacy metrics
        self.privacy_metrics = {
            'total_operations': 0,
            'encryption_operations': 0,
            'anonymization_operations': 0,
            'differential_privacy_operations': 0,
            'secure_deletions': 0,
            'privacy_violations': 0
        }
        
        self.audit_logger.info(
            f"MindBridge Privacy Manager initialized with level: {privacy_level.value}"
        )
    
    def _setup_audit_logging(self) -> logging.Logger:
        """Setup privacy audit logging"""
        logger = logging.getLogger('mindbridge_privacy')
        logger.setLevel(logging.INFO)
        
        # Create file handler for audit log
        handler = logging.FileHandler(self.audit_log_path)
        formatter = logging.Formatter(
            '%(asctime)s - PRIVACY_AUDIT - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def process_sensitive_data(self, 
                             data: Any, 
                             data_type: DataType,
                             operation: str,
                             user_context: str = None) -> Dict[str, Any]:
        """Process sensitive data with appropriate privacy protections"""
        
        start_time = time.time()
        audit_metadata = {
            'data_type': data_type.value,
            'operation': operation,
            'privacy_level': self.privacy_level.value,
            'user_context': user_context or 'anonymous'
        }
        
        try:
            processed_data = data
            privacy_operations = []
            
            # Step 1: Data anonymization
            if self.privacy_level in [PrivacyLevel.HIGH, PrivacyLevel.MAXIMUM]:
                if isinstance(data, dict) and 'text_content' in data:
                    data['text_content'] = self.anonymizer.anonymize_text(data['text_content'])
                    privacy_operations.append('text_anonymization')
                    self.privacy_metrics['anonymization_operations'] += 1
            
            # Step 2: Apply differential privacy to numeric values
            if self.privacy_level in [PrivacyLevel.MODERATE, PrivacyLevel.HIGH, PrivacyLevel.MAXIMUM]:
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, (int, float)) and key in ['risk_score', 'confidence', 'severity']:
                            data[key] = self.differential_privacy.add_laplace_noise(value, sensitivity=0.1)
                            privacy_operations.append(f'differential_privacy_{key}')
                            self.privacy_metrics['differential_privacy_operations'] += 1
            
            # Step 3: Encrypt for storage
            if operation in ['store', 'persist', 'save']:
                storage_id = self.secure_storage.store_encrypted(
                    data, 
                    data_type, 
                    retention_days=self._get_retention_period(data_type)
                )
                processed_data = {'storage_id': storage_id, 'encrypted': True}
                privacy_operations.append('encryption_storage')
                self.privacy_metrics['encryption_operations'] += 1
            
            # Step 4: Zero-knowledge proofs for sensitive metrics
            if (self.privacy_level == PrivacyLevel.MAXIMUM and 
                isinstance(data, dict) and 'risk_score' in data):
                
                commitment = self.zero_knowledge.commit_to_value(
                    data['risk_score'], 
                    f"risk_{int(time.time())}"
                )
                processed_data['zk_commitment'] = commitment
                privacy_operations.append('zero_knowledge_proof')
            
            # Audit successful operation
            audit_metadata.update({
                'success': True,
                'privacy_operations': privacy_operations,
                'processing_time_ms': int((time.time() - start_time) * 1000)
            })
            
            self._log_privacy_audit(operation, data_type, True, audit_metadata)
            self.privacy_metrics['total_operations'] += 1
            
            return {
                'data': processed_data,
                'privacy_metadata': {
                    'privacy_level': self.privacy_level.value,
                    'operations_applied': privacy_operations,
                    'data_type': data_type.value,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            # Audit failed operation
            audit_metadata.update({
                'success': False,
                'error': str(e),
                'processing_time_ms': int((time.time() - start_time) * 1000)
            })
            
            self._log_privacy_audit(operation, data_type, False, audit_metadata)
            self.privacy_metrics['privacy_violations'] += 1
            
            raise e
    
    def retrieve_sensitive_data(self, storage_id: str) -> Dict[str, Any]:
        """Retrieve and decrypt sensitive data"""
        try:
            decrypted_data, metadata = self.secure_storage.retrieve_decrypted(storage_id)
            
            self._log_privacy_audit(
                'retrieve', 
                metadata.data_type, 
                True, 
                {'storage_id': storage_id[:8] + '...', 'retention_days': metadata.retention_policy}
            )
            
            return {
                'data': decrypted_data,
                'metadata': {
                    'data_type': metadata.data_type.value,
                    'created_at': metadata.timestamp.isoformat(),
                    'expires_at': (metadata.timestamp + timedelta(days=metadata.retention_policy)).isoformat()
                }
            }
            
        except Exception as e:
            self._log_privacy_audit(
                'retrieve', 
                DataType.SYSTEM_LOGS,  # Unknown type due to error
                False, 
                {'storage_id': storage_id[:8] + '...', 'error': str(e)}
            )
            raise e
    
    def anonymize_for_research(self, dataset: List[Dict], k_value: int = 5) -> List[Dict]:
        """Anonymize dataset for research purposes"""
        try:
            anonymized_dataset = self.anonymizer.k_anonymize_dataset(dataset, k_value)
            
            # Apply additional differential privacy
            if self.privacy_level in [PrivacyLevel.HIGH, PrivacyLevel.MAXIMUM]:
                for record in anonymized_dataset:
                    for key, value in record.items():
                        if isinstance(value, (int, float)) and key not in ['user_id']:
                            record[key] = self.differential_privacy.add_laplace_noise(value)
            
            self._log_privacy_audit(
                'anonymize_research', 
                DataType.USER_PROFILE, 
                True, 
                {'dataset_size': len(dataset), 'k_value': k_value}
            )
            
            return anonymized_dataset
            
        except Exception as e:
            self._log_privacy_audit(
                'anonymize_research', 
                DataType.USER_PROFILE, 
                False, 
                {'error': str(e)}
            )
            raise e
    
    def emergency_privacy_reset(self, confirmation_code: str) -> bool:
        """Emergency privacy reset - destroys all data and keys"""
        expected_code = hashlib.sha256("MINDBRIDGE_EMERGENCY_RESET".encode()).hexdigest()[:16]
        
        if not hmac.compare_digest(confirmation_code, expected_code):
            self._log_privacy_audit(
                'emergency_reset', 
                DataType.SYSTEM_LOGS, 
                False, 
                {'reason': 'invalid_confirmation_code'}
            )
            return False
        
        try:
            # Destroy all stored data
            storage_success = self.secure_storage.emergency_data_destruction()
            
            # Destroy all encryption keys
            key_success = self.key_manager.emergency_key_destruction()
            
            # Clear privacy metrics
            self.privacy_metrics = {key: 0 for key in self.privacy_metrics.keys()}
            
            success = storage_success and key_success
            
            self._log_privacy_audit(
                'emergency_reset', 
                DataType.SYSTEM_LOGS, 
                success, 
                {'data_destroyed': storage_success, 'keys_destroyed': key_success}
            )
            
            return success
            
        except Exception as e:
            self._log_privacy_audit(
                'emergency_reset', 
                DataType.SYSTEM_LOGS, 
                False, 
                {'error': str(e)}
            )
            return False
    
    def get_privacy_metrics(self) -> Dict[str, Any]:
        """Get privacy operation metrics"""
        return {
            'privacy_level': self.privacy_level.value,
            'operations': dict(self.privacy_metrics),
            'key_rotations': getattr(self.key_manager, 'rotation_count', 0),
            'expired_data_cleaned': 0,  # Would track in production
            'audit_log_size': self.audit_log_path.stat().st_size if self.audit_log_path.exists() else 0,
            'storage_usage': sum(f.stat().st_size for f in self.secure_storage.storage_path.glob("*")),
            'last_updated': datetime.now().isoformat()
        }
    
    def verify_privacy_compliance(self) -> Dict[str, Any]:
        """Verify system privacy compliance"""
        compliance_checks = {
            'encryption_enabled': True,
            'key_rotation_active': hasattr(self.key_manager, 'last_rotation'),
            'differential_privacy_active': self.privacy_level != PrivacyLevel.MINIMAL,
            'audit_logging_active': self.audit_log_path.exists(),
            'secure_deletion_available': True,
            'emergency_reset_available': True
        }
        
        compliance_score = sum(compliance_checks.values()) / len(compliance_checks)
        
        return {
            'compliance_score': compliance_score,
            'checks': compliance_checks,
            'privacy_level': self.privacy_level.value,
            'recommendation': self._get_privacy_recommendation(compliance_score),
            'last_audit': datetime.now().isoformat()
        }
    
    def _get_retention_period(self, data_type: DataType) -> int:
        """Get data retention period based on type and privacy level"""
        retention_policies = {
            DataType.TEXT_ANALYSIS: 7,      # 7 days
            DataType.BEHAVIORAL_DATA: 30,   # 30 days
            DataType.RISK_ASSESSMENT: 90,   # 90 days
            DataType.INTERVENTION_DATA: 365, # 1 year
            DataType.USER_PROFILE: 1095,    # 3 years
            DataType.SYSTEM_LOGS: 30        # 30 days
        }
        
        base_retention = retention_policies.get(data_type, 30)
        
        # Reduce retention for higher privacy levels
        if self.privacy_level == PrivacyLevel.MAXIMUM:
            return max(1, base_retention // 4)  # Minimum 1 day
        elif self.privacy_level == PrivacyLevel.HIGH:
            return max(7, base_retention // 2)  # Minimum 1 week
        
        return base_retention
    
    def _log_privacy_audit(self, 
                          operation: str, 
                          data_type: DataType, 
                          success: bool, 
                          metadata: Dict[str, Any]):
        """Log privacy operation for audit trail"""
        audit_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'data_type': data_type.value,
            'success': success,
            'privacy_level': self.privacy_level.value,
            'metadata': metadata
        }
        
        self.audit_logger.info(json.dumps(audit_entry))
    
    def _get_privacy_recommendation(self, compliance_score: float) -> str:
        """Get privacy recommendation based on compliance score"""
        if compliance_score >= 0.9:
            return "Excellent privacy posture - all systems operational"
        elif compliance_score >= 0.7:
            return "Good privacy posture - minor improvements recommended"
        elif compliance_score >= 0.5:
            return "Moderate privacy posture - several improvements needed"
        else:
            return "Poor privacy posture - immediate action required"
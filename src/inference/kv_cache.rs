//! Machine Intelligence Node - Key-Value Cache
//!
//! Implements a high-performance key-value caching system for AI inference optimization.
//!
//! Author: Machine Intelligence Node Development Team

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use std::thread;
use std::hash::Hash;

/// Represents a cache entry with expiration tracking.
struct CacheEntry<V> {
    value: V,
    expiration: Option<Instant>,
}

/// A thread-safe key-value cache with TTL (Time-to-Live) support.
pub struct KVCache<K, V> 
where 
    K: Eq + Hash + Clone, 
    V: Clone 
{
    store: Arc<RwLock<HashMap<K, CacheEntry<V>>>>,
}

impl<K, V> KVCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Creates a new KVCache instance.
    pub fn new() -> Self {
        KVCache {
            store: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Inserts a key-value pair into the cache with an optional TTL.
    ///
    /// # Arguments
    /// * `key` - The key to store the value under.
    /// * `value` - The value to store.
    /// * `ttl` - Optional expiration time in seconds.
    pub fn insert(&self, key: K, value: V, ttl: Option<u64>) {
        let expiration = ttl.map(|t| Instant::now() + Duration::from_secs(t));
        let mut store = self.store.write().unwrap();
        store.insert(key, CacheEntry { value, expiration });
    }

    /// Retrieves a value from the cache.
    ///
    /// # Arguments
    /// * `key` - The key to retrieve.
    ///
    /// # Returns
    /// * `Option<V>` - The cached value if present and not expired.
    pub fn get(&self, key: &K) -> Option<V> {
        let store = self.store.read().unwrap();
        if let Some(entry) = store.get(key) {
            if let Some(expiration) = entry.expiration {
                if Instant::now() > expiration {
                    return None;
                }
            }
            return Some(entry.value.clone());
        }
        None
    }

    /// Removes a key from the cache.
    ///
    /// # Arguments
    /// * `key` - The key to remove.
    pub fn remove(&self, key: &K) {
        let mut store = self.store.write().unwrap();
        store.remove(key);
    }

    /// Clears all expired entries from the cache.
    pub fn cleanup(&self) {
        let mut store = self.store.write().unwrap();
        store.retain(|_, entry| {
            entry.expiration.map_or(true, |exp| Instant::now() < exp)
        });
    }
}

/// Background cleanup task for expired cache entries.
pub fn start_cleanup_task<K, V>(cache: Arc<KVCache<K, V>>, interval: u64)
where
    K: Eq + Hash + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    thread::spawn(move || {
        loop {
            thread::sleep(Duration::from_secs(interval));
            cache.cleanup();
        }
    });
}

/// Example Usage
fn main() {
    let cache = Arc::new(KVCache::<String, String>::new());

    // Start background cleanup task
    start_cleanup_task(cache.clone(), 60);

    // Insert data into the cache
    cache.insert("model_output".to_string(), "prediction_123".to_string(), Some(30));
    
    // Retrieve data
    if let Some(value) = cache.get(&"model_output".to_string()) {
        println!("Cached Value: {}", value);
    } else {
        println!("Cache Miss");
    }
}

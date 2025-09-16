# calculator.py
import math

class WeaviateStorageCalculator:
	"""
	Handles all storage calculation logic for Weaviate.
	"""

	def calculate_storage(
		self,
		num_objects,
		vector_dimensions,
		avg_object_size_bytes, 
		quantization="none",
		filterable_index_factor=0.1700,
		searchable_index_factor=0.0500,
	):
		"""
		Calculates Weaviate disk storage with accurate handling of quantization.
		With quantization, both the original uncompressed vectors and the
		compressed HNSW index are stored on disk.
		"""
		# --- Base Components ---
		# 1. Storage for raw object properties (excluding any vectors)
		total_object_properties_storage = num_objects * avg_object_size_bytes

		# 2. Storage for the full, uncompressed float32 vectors
		uncompressed_vector_storage = num_objects * vector_dimensions * 4

		# 3. Inverted Index Storage
		base_for_inverted_index = total_object_properties_storage
		filterable_storage = base_for_inverted_index * filterable_index_factor
		searchable_storage = base_for_inverted_index * searchable_index_factor

		# --- HNSW Index Component ---
		# This is the storage for the graph index itself.
		hnsw_index_storage = 0
		if quantization == 'none':
			# Without quantization, the HNSW index IS the uncompressed vector store.
			# We avoid double-counting by setting the separate uncompressed store to 0.
			hnsw_index_storage = uncompressed_vector_storage
			uncompressed_vector_storage = 0
		else:
			# With quantization, the HNSW index is built from smaller, compressed vectors.
			compressed_bytes_per_object = 0
			if quantization == 'sq':
				# SQ is 8-bits (1 byte) per dimension.
				compressed_bytes_per_object = vector_dimensions * 1
			elif quantization == 'pq':
				# Based on official 768d -> 128 byte example.
				if vector_dimensions == 768:
					compressed_bytes_per_object = 128
				else:
					# General formula for other dimensions.
					compressed_bytes_per_object = math.ceil(vector_dimensions / 12) * 2
			elif quantization == 'rq':
				#Rounds dims up to nearest 64, then uses 1 byte.
				rounded_dims = math.ceil(vector_dimensions / 64) * 64
				compressed_bytes_per_object = rounded_dims * 1
			elif quantization == 'bq':
				# BQ is 1 bit per dimension.
				compressed_bytes_per_object = math.ceil(vector_dimensions / 8)

			hnsw_index_storage = num_objects * compressed_bytes_per_object

		# --- Total Calculation ---
		# The total is the sum of all components stored on disk.
		total_storage_bytes = (
			total_object_properties_storage
			+ filterable_storage
			+ searchable_storage
			+ uncompressed_vector_storage # This is non-zero ONLY if quantization is enabled.
			+ hnsw_index_storage
		)

		return {
			"object_properties_storage_bytes": total_object_properties_storage,
			"filterable_storage_bytes": filterable_storage,
			"searchable_storage_bytes": searchable_storage,
			"uncompressed_vector_storage_bytes": uncompressed_vector_storage,
			"hnsw_index_storage_bytes": hnsw_index_storage,
			"total_storage_bytes": total_storage_bytes,
		}

	def extrapolate_from_sample(self, sample_objects, sample_storage_bytes, target_objects):
		"""
		Extrapolate storage requirements from a sample dataset.
		"""
		if sample_objects == 0:
			return {
				"linear_extrapolation_gb": 0,
				"conservative_extrapolation_gb": 0,
				"storage_per_object": 0
			}

		storage_per_object = sample_storage_bytes / sample_objects
		linear_storage = storage_per_object * target_objects

		conservative_storage = linear_storage
		if target_objects > sample_objects:
			scale_factor = target_objects / sample_objects
			# Add a modest overhead for indexing inefficiencies at scale.
			overhead_factor = 1 + (math.log10(scale_factor) * 0.025)
			conservative_storage = linear_storage * overhead_factor

		return {
			"linear_extrapolation_gb": linear_storage / (1024**3),
			"conservative_extrapolation_gb": conservative_storage / (1024**3),
			"storage_per_object": storage_per_object
		}

def format_bytes(bytes_value):
	"""Format bytes to a human-readable string."""
	if bytes_value < 1024:
		return f"{bytes_value:.2f} B"
	elif bytes_value < 1024**2:
		return f"{bytes_value / 1024:.2f} KB"
	elif bytes_value < 1024**3:
		return f"{bytes_value / (1024**2):.2f} MB"
	else:
		return f"{bytes_value / (1024**3):.2f} GB"

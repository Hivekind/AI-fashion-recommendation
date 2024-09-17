#!/bin/bash

# Check if the database exists
DB_EXISTS=$(psql -U postgres -tAc "SELECT 1 FROM pg_database WHERE datname='fashion_data';")

# Early return if the database already exists
if [ "$DB_EXISTS" = '1' ]; then
  echo "Database 'fashion_data' already exists. Skipping initialization."
  exit 0
fi

# Proceed with initialization if the database does not exist
echo "Database 'fashion_data' does not exist. Initializing database..."

# Create the database
psql -U postgres -c "CREATE DATABASE fashion_data;"

# Install the pgvector extension in the new database
psql -U postgres -d fashion_data -c "CREATE EXTENSION vector;"

# Restore the dump file using 4 parallel jobs
pg_restore -U postgres -d fashion_data -j 4 /init-scripts/fashion_data.dump

echo "Database 'fashion_data' initialized."

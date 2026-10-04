# Pinecone configuration

Create a replacement API key in the correct Pinecone project and delete the
exposed key in the Pinecone console. Removing a key from Git does not revoke it.
Keep the replacement in your environment or deployment secret store.

## Docker Compose (docker-work branch)

Copy `.env.example` to `.env` in the repository root and fill in the new key:

```dotenv
PINECONE_API_KEY=your-new-key
```

Then run `docker compose up --build`. Compose passes the key to the searcher at
runtime and reports an error if it is missing or empty. The frontend and
extraction service do not need Pinecone credentials. After a rotation, recreate
the searcher container with `docker compose up -d --force-recreate searcher`.

`.env` files are ignored by Git. Keep them out of container images and archives.
The `.env.example` file contains no credential.

## Direct Python execution (main branch or searcher)

The feature upload, query, Flask application, and searcher read
`PINECONE_API_KEY` from the process environment. They report a clear error if the
variable is unset, empty, or whitespace-only. Python does not load `.env` files
automatically. In PowerShell, enter the new key without recording it as a
literal in command history:

```powershell
$pineconeSecret = Read-Host "New Pinecone API key" -AsSecureString
$env:PINECONE_API_KEY = [System.Net.NetworkCredential]::new("", $pineconeSecret).Password
```

Run the desired Python entry point from that same shell. In production, inject
the variable through the deployment's secret configuration.

## After a history purge

Both published branches must use the cleaned history. Collaborators should
clone the repository again and reapply their local changes after removing any
old credentials. Merging or pushing old history can reintroduce the key. Clean
other clones, archives, and deployed copies too. GitHub may retain unreachable
commit views and caches; follow its sensitive-data-removal process for those
copies. Verify key deletion in Pinecone independently of the Git cleanup.

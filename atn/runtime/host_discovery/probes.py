"""Concrete host probes for common tools and services."""

from __future__ import annotations

import os
import re
from pathlib import Path

from .probe import HostProbe, ProbeResult


# ── GitHub CLI ────────────────────────────────────────────────────────

class GitHubProbe(HostProbe):
    id = "host_github"
    name = "GitHub"
    provider = "github"
    description = "Code hosting, pull requests, CI/CD pipelines, and collaboration via the gh CLI."
    tags = ["dev", "cloud"]
    install_hint = "winget install GitHub.cli"
    auth_hint = "Run: gh auth login"
    url = "https://cli.github.com"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("gh"):
            return r
        r.available = True

        code, out, _ = await self.run_cmd("gh", "--version")
        if code == 0:
            m = re.search(r"(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        code, out, err = await self.run_cmd("gh", "auth", "status")
        combined = out + err
        if code == 0 or "Logged in" in combined:
            r.authenticated = True
            r.auth_method = "cli"
            m = re.search(r"account\s+(\S+)", combined)
            if m:
                r.account = m.group(1)
            r.capabilities = ["repos", "issues", "pull_requests", "gists", "actions", "packages"]
            r.detail = "gh CLI authenticated"

        if not r.authenticated and os.environ.get("GITHUB_TOKEN"):
            r.authenticated = True
            r.auth_method = "env_var"
            r.capabilities = ["api_access"]
            r.detail = "GITHUB_TOKEN set"

        return r


# ── AWS CLI ───────────────────────────────────────────────────────────

class AwsProbe(HostProbe):
    id = "host_aws"
    name = "AWS"
    provider = "amazon"
    description = "Amazon Web Services — S3 storage, EC2 compute, Lambda functions, and 200+ cloud services."
    tags = ["cloud", "devops"]
    install_hint = "winget install Amazon.AWSCLI"
    auth_hint = "Run: aws configure"
    url = "https://aws.amazon.com/cli/"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("aws"):
            return r
        r.available = True

        code, out, _ = await self.run_cmd("aws", "--version")
        if code == 0:
            m = re.search(r"aws-cli/(\S+)", out)
            if m:
                r.version = m.group(1)

        code, out, _ = await self.run_cmd("aws", "sts", "get-caller-identity")
        if code == 0:
            r.authenticated = True
            r.auth_method = "cli"
            import json as _json
            try:
                data = _json.loads(out)
                r.account = data.get("Account", "")
                arn = data.get("Arn", "")
                if arn:
                    r.detail = f"identity: {arn}"
            except Exception:
                pass
            r.capabilities = ["s3", "ec2", "lambda", "iam", "cloudformation",
                              "dynamodb", "sqs", "sns"]

        if not r.authenticated:
            creds = Path.home() / ".aws" / "credentials"
            if creds.exists():
                r.auth_method = "config_file"
                r.detail = "~/.aws/credentials found (not validated)"
                r.capabilities = ["s3", "ec2", "lambda", "iam"]

        return r


# ── Google Cloud ──────────────────────────────────────────────────────

class GcpProbe(HostProbe):
    id = "host_gcp"
    name = "Google Cloud"
    provider = "google"
    description = "Google Cloud Platform — BigQuery analytics, Cloud Run containers, GKE clusters, and storage."
    tags = ["cloud", "devops"]
    install_hint = "winget install Google.CloudSDK"
    auth_hint = "Run: gcloud auth login"
    url = "https://cloud.google.com/sdk"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("gcloud"):
            return r
        r.available = True

        code, out, _ = await self.run_cmd("gcloud", "--version")
        if code == 0:
            m = re.search(r"Google Cloud SDK\s+(\S+)", out)
            if m:
                r.version = m.group(1)

        code, out, _ = await self.run_cmd("gcloud", "auth", "list",
                                          "--filter=status:ACTIVE",
                                          "--format=value(account)")
        if code == 0 and out.strip():
            r.authenticated = True
            r.auth_method = "cli"
            r.account = out.strip().splitlines()[0]
            r.capabilities = ["compute", "storage", "bigquery", "pubsub",
                              "cloud_run", "cloud_functions", "gke"]

        code, proj_out, _ = await self.run_cmd("gcloud", "config", "get-value", "project")
        if code == 0 and proj_out.strip():
            r.detail = f"project: {proj_out.strip()}"

        return r


# ── Azure ─────────────────────────────────────────────────────────────

class AzureProbe(HostProbe):
    id = "host_azure"
    name = "Azure"
    provider = "microsoft"
    description = "Microsoft Azure — VMs, App Service, Functions, Cosmos DB, and enterprise identity."
    tags = ["cloud", "devops"]
    install_hint = "winget install Microsoft.AzureCLI"
    auth_hint = "Run: az login"
    url = "https://learn.microsoft.com/cli/azure/"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("az"):
            return r
        r.available = True

        code, out, _ = await self.run_cmd("az", "version", "-o", "tsv")
        if code == 0:
            m = re.search(r"(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        code, out, _ = await self.run_cmd("az", "account", "show", "-o", "json")
        if code == 0:
            r.authenticated = True
            r.auth_method = "cli"
            import json as _json
            try:
                data = _json.loads(out)
                r.account = data.get("user", {}).get("name", "")
                r.detail = f"subscription: {data.get('name', '')}"
            except Exception:
                pass
            r.capabilities = ["vms", "storage", "functions", "cosmos_db",
                              "aks", "app_service"]

        return r


# ── Docker ────────────────────────────────────────────────────────────

class DockerProbe(HostProbe):
    id = "host_docker"
    name = "Docker"
    provider = "docker"
    description = "Container runtime for building, shipping, and running applications in isolated environments."
    tags = ["dev", "devops"]
    install_hint = "Download Docker Desktop"
    auth_hint = "Start Docker Desktop"
    url = "https://www.docker.com/products/docker-desktop/"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("docker"):
            return r
        r.available = True

        code, out, _ = await self.run_cmd("docker", "--version")
        if code == 0:
            m = re.search(r"(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        code, out, _ = await self.run_cmd("docker", "info", "--format", "{{.ServerVersion}}")
        if code == 0 and out.strip():
            r.authenticated = True
            r.auth_method = "none"
            r.detail = f"server {out.strip()}"
            r.capabilities = ["containers", "images", "volumes", "networks", "compose"]
        else:
            r.detail = "installed but daemon not running"

        return r


# ── Kubernetes ────────────────────────────────────────────────────────

class KubernetesProbe(HostProbe):
    id = "host_kubernetes"
    name = "Kubernetes"
    provider = "kubernetes"
    description = "Container orchestration for deploying, scaling, and managing containerized workloads."
    tags = ["devops", "cloud"]
    install_hint = "winget install Kubernetes.kubectl"
    auth_hint = "Configure: kubectl config set-context"
    url = "https://kubernetes.io/docs/tasks/tools/"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("kubectl"):
            return r
        r.available = True

        code, out, _ = await self.run_cmd("kubectl", "version", "--client", "-o", "json")
        if code == 0:
            import json as _json
            try:
                data = _json.loads(out)
                v = data.get("clientVersion", {}).get("gitVersion", "")
                if v:
                    r.version = v.lstrip("v")
            except Exception:
                pass

        kubeconfig = Path.home() / ".kube" / "config"
        if kubeconfig.exists():
            r.auth_method = "config_file"
            code, ctx, _ = await self.run_cmd("kubectl", "config", "current-context")
            if code == 0 and ctx.strip():
                r.authenticated = True
                r.account = ctx.strip()
                r.detail = f"context: {ctx.strip()}"
                r.capabilities = ["pods", "deployments", "services",
                                  "configmaps", "secrets", "namespaces"]

        return r


# ── Netlify ───────────────────────────────────────────────────────────

class NetlifyProbe(HostProbe):
    id = "host_netlify"
    name = "Netlify"
    provider = "netlify"
    description = "Web hosting platform with instant deploys, serverless functions, and edge networking."
    tags = ["dev", "cloud"]
    install_hint = "npm install -g netlify-cli"
    auth_hint = "Run: netlify login"
    url = "https://docs.netlify.com/cli/get-started/"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("netlify"):
            return r
        r.available = True

        code, out, _ = await self.run_cmd("netlify", "--version")
        if code == 0:
            m = re.search(r"(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        code, out, err = await self.run_cmd("netlify", "status")
        combined = out + err
        if code == 0 and "Logged in" in combined:
            r.authenticated = True
            r.auth_method = "cli"
            r.capabilities = ["deploy", "sites", "functions", "forms", "dns"]
            m = re.search(r"Email:\s+(\S+)", combined)
            if m:
                r.account = m.group(1)

        return r


# ── Vercel ────────────────────────────────────────────────────────────

class VercelProbe(HostProbe):
    id = "host_vercel"
    name = "Vercel"
    provider = "vercel"
    description = "Frontend cloud platform — instant previews, serverless functions, and global edge deployments."
    tags = ["dev", "cloud"]
    install_hint = "npm install -g vercel"
    auth_hint = "Run: vercel login"
    url = "https://vercel.com/docs/cli"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("vercel"):
            return r
        r.available = True

        code, out, _ = await self.run_cmd("vercel", "--version")
        if code == 0:
            m = re.search(r"(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        code, out, _ = await self.run_cmd("vercel", "whoami")
        if code == 0 and out.strip():
            r.authenticated = True
            r.auth_method = "cli"
            r.account = out.strip()
            r.capabilities = ["deploy", "domains", "env", "projects"]

        return r


# ── Firebase ──────────────────────────────────────────────────────────

class FirebaseProbe(HostProbe):
    id = "host_firebase"
    name = "Firebase"
    provider = "google"
    description = "Google's app platform — realtime database, auth, hosting, cloud functions, and push notifications."
    tags = ["dev", "cloud"]
    install_hint = "npm install -g firebase-tools"
    auth_hint = "Run: firebase login"
    url = "https://firebase.google.com/docs/cli"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("firebase"):
            return r
        r.available = True

        code, out, _ = await self.run_cmd("firebase", "--version")
        if code == 0:
            m = re.search(r"(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        code, out, _ = await self.run_cmd("firebase", "login:list")
        if code == 0 and out.strip() and "No authorized" not in out:
            r.authenticated = True
            r.auth_method = "cli"
            r.capabilities = ["hosting", "firestore", "auth", "functions",
                              "storage", "realtime_db"]
            # Try to extract email
            for line in out.splitlines():
                if "@" in line:
                    r.account = line.strip().split()[-1]
                    break

        return r


# ── Terraform ─────────────────────────────────────────────────────────

class TerraformProbe(HostProbe):
    id = "host_terraform"
    name = "Terraform"
    provider = "hashicorp"
    description = "Infrastructure as code — provision and manage cloud resources with declarative config files."
    tags = ["devops", "cloud"]
    install_hint = "winget install Hashicorp.Terraform"
    auth_hint = ""
    url = "https://www.terraform.io"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("terraform"):
            return r
        r.available = True
        r.auth_method = "none"

        code, out, _ = await self.run_cmd("terraform", "--version")
        if code == 0:
            m = re.search(r"v(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)
            r.authenticated = True
            r.capabilities = ["plan", "apply", "destroy", "state", "import"]

        return r


# ── SSH keys ──────────────────────────────────────────────────────────

class SshProbe(HostProbe):
    id = "host_ssh"
    name = "SSH"
    provider = "openssh"
    description = "Secure shell for remote access, tunneling, and encrypted file transfer."
    tags = ["dev", "security"]
    install_hint = "Included with Windows (Settings > Apps > Optional Features)"
    auth_hint = "Run: ssh-keygen -t ed25519"
    url = "https://www.openssh.com"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("ssh"):
            return r
        r.available = True

        code, out, err = await self.run_cmd("ssh", "-V")
        combined = out + err  # ssh -V prints to stderr
        m = re.search(r"OpenSSH[_\s]+(\S+)", combined)
        if m:
            r.version = m.group(1)

        ssh_dir = Path.home() / ".ssh"
        if ssh_dir.exists():
            keys = [f.stem for f in ssh_dir.iterdir()
                    if f.suffix == ".pub" and f.is_file()]
            if keys:
                r.authenticated = True
                r.auth_method = "config_file"
                r.detail = f"keys: {', '.join(keys)}"
                r.capabilities = ["remote_access", "git_ssh", "tunneling", "scp"]

        return r


# ── Git ───────────────────────────────────────────────────────────────

class GitProbe(HostProbe):
    id = "host_git"
    name = "Git"
    provider = "git"
    description = "Distributed version control — track changes, branch, merge, and collaborate on code."
    tags = ["dev"]
    install_hint = "winget install Git.Git"
    auth_hint = "Run: git config --global user.name \"Your Name\""
    url = "https://git-scm.com"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("git"):
            return r
        r.available = True

        code, out, _ = await self.run_cmd("git", "--version")
        if code == 0:
            m = re.search(r"(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        code, user, _ = await self.run_cmd("git", "config", "--global", "user.name")
        code2, email, _ = await self.run_cmd("git", "config", "--global", "user.email")
        if code == 0 and user.strip():
            r.authenticated = True
            r.auth_method = "config_file"
            r.account = email.strip() if code2 == 0 else user.strip()
            r.detail = f"{user.strip()} <{email.strip()}>" if code2 == 0 else user.strip()
            r.capabilities = ["clone", "commit", "push", "pull", "branch"]

        return r


# ── Node.js ───────────────────────────────────────────────────────────

class NodeProbe(HostProbe):
    id = "host_node"
    name = "Node.js"
    provider = "nodejs"
    description = "JavaScript runtime for server-side applications, build tools, and package management via npm."
    tags = ["dev"]
    install_hint = "winget install OpenJS.NodeJS.LTS"
    auth_hint = ""
    url = "https://nodejs.org"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("node"):
            return r
        r.available = True
        r.authenticated = True
        r.auth_method = "none"

        code, out, _ = await self.run_cmd("node", "--version")
        if code == 0:
            r.version = out.strip().lstrip("v")

        r.capabilities = ["runtime"]

        # Check npm
        if self.has_command("npm"):
            code, out, _ = await self.run_cmd("npm", "--version")
            if code == 0:
                r.capabilities.append("npm")
                r.detail = f"npm {out.strip()}"

            code, out, _ = await self.run_cmd("npm", "whoami")
            if code == 0 and out.strip():
                r.account = out.strip()
                r.capabilities.append("npm_publish")

        return r


# ── Python ────────────────────────────────────────────────────────────

class PythonProbe(HostProbe):
    id = "host_python"
    name = "Python"
    provider = "python"
    description = "General-purpose programming language — scripting, data science, web backends, and automation."
    tags = ["dev"]
    install_hint = "winget install Python.Python.3.12"
    auth_hint = ""
    url = "https://www.python.org"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("python") and not self.has_command("python3"):
            return r
        r.available = True
        r.authenticated = True
        r.auth_method = "none"

        cmd = "python3" if self.has_command("python3") else "python"
        code, out, _ = await self.run_cmd(cmd, "--version")
        if code == 0:
            m = re.search(r"(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        r.capabilities = ["runtime"]

        if self.has_command("pip") or self.has_command("pip3"):
            r.capabilities.append("pip")

        return r


# ── Anthropic API key ─────────────────────────────────────────────────

class AnthropicKeyProbe(HostProbe):
    id = "host_anthropic"
    name = "Anthropic API"
    provider = "anthropic"
    description = "Claude language models — conversation, analysis, code generation, and tool use."
    tags = ["ai"]
    install_hint = ""
    auth_hint = "Set ANTHROPIC_API_KEY environment variable"
    url = "https://console.anthropic.com"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            return r
        r.available = True
        r.authenticated = True
        r.auth_method = "env_var"
        r.detail = "key set (env)"
        r.capabilities = ["chat", "completions", "embeddings"]
        return r


# ── OpenAI API key ────────────────────────────────────────────────────

class OpenAiKeyProbe(HostProbe):
    id = "host_openai"
    name = "OpenAI API"
    provider = "openai"
    description = "GPT models, DALL-E image generation, Whisper transcription, and text-to-speech."
    tags = ["ai"]
    install_hint = ""
    auth_hint = "Set OPENAI_API_KEY environment variable"
    url = "https://platform.openai.com/api-keys"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            return r
        r.available = True
        r.authenticated = True
        r.auth_method = "env_var"
        r.detail = "key set (env)"
        r.capabilities = ["chat", "completions", "embeddings", "images", "tts"]
        return r


# ── Rust / Cargo ──────────────────────────────────────────────────────

class RustProbe(HostProbe):
    id = "host_rust"
    name = "Rust"
    provider = "rust"
    description = "Systems programming language with memory safety — ideal for performance-critical applications."
    tags = ["dev"]
    install_hint = "Download rustup from rustup.rs"
    auth_hint = ""
    url = "https://rustup.rs"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("rustc"):
            return r
        r.available = True
        r.authenticated = True
        r.auth_method = "none"

        code, out, _ = await self.run_cmd("rustc", "--version")
        if code == 0:
            m = re.search(r"(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        r.capabilities = ["compiler"]
        if self.has_command("cargo"):
            r.capabilities.append("cargo")
            r.detail = "cargo available"

        return r


# ── Go ────────────────────────────────────────────────────────────────

class GoProbe(HostProbe):
    id = "host_go"
    name = "Go"
    provider = "golang"
    description = "Compiled language built for concurrency — great for CLIs, APIs, and distributed systems."
    tags = ["dev"]
    install_hint = "winget install GoLang.Go"
    auth_hint = ""
    url = "https://go.dev"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("go"):
            return r
        r.available = True
        r.authenticated = True
        r.auth_method = "none"

        code, out, _ = await self.run_cmd("go", "version")
        if code == 0:
            m = re.search(r"go(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        r.capabilities = ["compiler", "modules"]
        return r


# ── Java / JDK ────────────────────────────────────────────────────────

class JavaProbe(HostProbe):
    id = "host_java"
    name = "Java"
    provider = "oracle"
    description = "Enterprise runtime — Spring backends, Android apps, big data with Hadoop and Spark."
    tags = ["dev"]
    install_hint = "winget install Microsoft.OpenJDK.21"
    auth_hint = ""
    url = "https://adoptium.net"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("java"):
            return r
        r.available = True
        r.authenticated = True
        r.auth_method = "none"

        code, out, err = await self.run_cmd("java", "-version")
        combined = out + err
        m = re.search(r'"(\d+[\.\d+]*)"', combined)
        if m:
            r.version = m.group(1)

        r.capabilities = ["runtime"]
        if self.has_command("mvn"):
            r.capabilities.append("maven")
        if self.has_command("gradle"):
            r.capabilities.append("gradle")
        return r


# ── PostgreSQL ────────────────────────────────────────────────────────

class PostgresProbe(HostProbe):
    id = "host_postgres"
    name = "PostgreSQL"
    provider = "postgresql"
    description = "Advanced open-source relational database with JSON support and full-text search."
    tags = ["data", "dev"]
    install_hint = "winget install PostgreSQL.PostgreSQL"
    auth_hint = "Set PGHOST or DATABASE_URL environment variable"
    url = "https://www.postgresql.org"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("psql"):
            return r
        r.available = True

        code, out, _ = await self.run_cmd("psql", "--version")
        if code == 0:
            m = re.search(r"(\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        r.capabilities = ["sql", "json", "full_text_search"]

        if os.environ.get("PGHOST") or os.environ.get("DATABASE_URL"):
            r.authenticated = True
            r.auth_method = "env_var"
            r.detail = "connection env vars set"

        return r


# ── Redis ─────────────────────────────────────────────────────────────

class RedisProbe(HostProbe):
    id = "host_redis"
    name = "Redis"
    provider = "redis"
    description = "In-memory data store — caching, message queues, real-time leaderboards, and session management."
    tags = ["data", "dev"]
    install_hint = "winget install Redis.Redis"
    auth_hint = "Start Redis server"
    url = "https://redis.io"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("redis-cli"):
            return r
        r.available = True

        code, out, _ = await self.run_cmd("redis-cli", "--version")
        if code == 0:
            m = re.search(r"(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        code, out, _ = await self.run_cmd("redis-cli", "ping")
        if code == 0 and "PONG" in out:
            r.authenticated = True
            r.auth_method = "none"
            r.detail = "server responding"
            r.capabilities = ["cache", "pubsub", "streams", "search"]

        return r


# ── MongoDB ───────────────────────────────────────────────────────────

class MongoProbe(HostProbe):
    id = "host_mongo"
    name = "MongoDB"
    provider = "mongodb"
    description = "Document database for flexible schemas — great for rapid prototyping and content-heavy apps."
    tags = ["data", "dev"]
    install_hint = "winget install MongoDB.Shell"
    auth_hint = "Set MONGODB_URI environment variable"
    url = "https://www.mongodb.com"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("mongosh") and not self.has_command("mongo"):
            return r
        r.available = True

        cmd = "mongosh" if self.has_command("mongosh") else "mongo"
        code, out, _ = await self.run_cmd(cmd, "--version")
        if code == 0:
            m = re.search(r"(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        r.capabilities = ["documents", "aggregation", "indexes"]

        if os.environ.get("MONGODB_URI") or os.environ.get("MONGO_URL"):
            r.authenticated = True
            r.auth_method = "env_var"
            r.detail = "connection URI set"

        return r


# ── SQLite ────────────────────────────────────────────────────────────

class SqliteProbe(HostProbe):
    id = "host_sqlite"
    name = "SQLite"
    provider = "sqlite"
    description = "Embedded SQL database — zero config, single-file storage for local apps and prototyping."
    tags = ["data", "dev"]
    install_hint = "Bundled with Python; or winget install SQLite.SQLite"
    auth_hint = ""
    url = "https://www.sqlite.org"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("sqlite3"):
            return r
        r.available = True
        r.authenticated = True
        r.auth_method = "none"

        code, out, _ = await self.run_cmd("sqlite3", "--version")
        if code == 0:
            m = re.search(r"(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        r.capabilities = ["sql", "full_text_search"]
        return r


# ── Ollama (local LLMs) ──────────────────────────────────────────────

class OllamaProbe(HostProbe):
    id = "host_ollama"
    name = "Ollama"
    provider = "ollama"
    description = "Run large language models locally — Llama, Mistral, CodeLlama, and more. Private and offline."
    tags = ["ai"]
    install_hint = "winget install Ollama.Ollama"
    auth_hint = "Run: ollama pull llama3"
    url = "https://ollama.com"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("ollama"):
            return r
        r.available = True

        code, out, _ = await self.run_cmd("ollama", "--version")
        if code == 0:
            m = re.search(r"(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        code, out, _ = await self.run_cmd("ollama", "list")
        if code == 0:
            r.authenticated = True
            r.auth_method = "none"
            models = [l.split()[0] for l in out.strip().splitlines()[1:] if l.strip()]
            r.capabilities = models[:8] if models else ["inference"]
            r.detail = f"{len(models)} model{'s' if len(models) != 1 else ''} installed"

        return r


# ── HuggingFace CLI ───────────────────────────────────────────────────

class HuggingFaceProbe(HostProbe):
    id = "host_huggingface"
    name = "Hugging Face"
    provider = "huggingface"
    description = "ML model hub — download and deploy thousands of pre-trained models for NLP, vision, and audio."
    tags = ["ai", "data"]
    install_hint = "pip install huggingface_hub"
    auth_hint = "Run: huggingface-cli login, or set HF_TOKEN"
    url = "https://huggingface.co"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        key = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        has_cli = self.has_command("huggingface-cli")

        if not has_cli and not key:
            return r
        r.available = has_cli or bool(key)

        if has_cli:
            code, out, _ = await self.run_cmd("huggingface-cli", "version")
            if code == 0:
                m = re.search(r"(\d+\.\d+\.\d+)", out)
                if m:
                    r.version = m.group(1)

            code, out, _ = await self.run_cmd("huggingface-cli", "whoami")
            if code == 0 and out.strip() and "Not logged in" not in out:
                r.authenticated = True
                r.auth_method = "cli"
                r.account = out.strip().splitlines()[0]

        if not r.authenticated and key:
            r.authenticated = True
            r.auth_method = "env_var"
            r.detail = "HF token set"

        r.capabilities = ["models", "datasets", "spaces", "inference"]
        return r


# ── GPG ───────────────────────────────────────────────────────────────

class GpgProbe(HostProbe):
    id = "host_gpg"
    name = "GPG"
    provider = "gnupg"
    description = "Encryption and signing — verify commits, encrypt files, and manage cryptographic keys."
    tags = ["security", "dev"]
    install_hint = "winget install GnuPG.GnuPG"
    auth_hint = "Run: gpg --gen-key"
    url = "https://gnupg.org"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("gpg"):
            return r
        r.available = True

        code, out, _ = await self.run_cmd("gpg", "--version")
        if code == 0:
            m = re.search(r"(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        code, out, _ = await self.run_cmd("gpg", "--list-secret-keys", "--keyid-format", "long")
        if code == 0 and out.strip():
            r.authenticated = True
            r.auth_method = "config_file"
            keys = re.findall(r"sec\s+\S+/(\w+)", out)
            r.detail = f"{len(keys)} secret key{'s' if len(keys) != 1 else ''}"
            r.capabilities = ["sign", "encrypt", "verify", "git_signing"]

        return r


# ── Foundry (Ethereum) ───────────────────────────────────────────────

class FoundryProbe(HostProbe):
    id = "host_foundry"
    name = "Foundry"
    provider = "ethereum"
    description = "Ethereum development toolkit — compile Solidity, run tests, deploy contracts, and trace transactions."
    tags = ["web3", "dev"]
    install_hint = "curl -L https://foundry.paradigm.xyz | bash && foundryup"
    auth_hint = ""
    url = "https://book.getfoundry.sh"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("forge"):
            return r
        r.available = True
        r.authenticated = True
        r.auth_method = "none"

        code, out, _ = await self.run_cmd("forge", "--version")
        if code == 0:
            m = re.search(r"(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        r.capabilities = ["compile", "test", "deploy", "verify", "cast"]
        return r


# ── Etherscan API ────────────────────────────────────────────────────

class EtherscanProbe(HostProbe):
    id = "host_etherscan"
    name = "Etherscan API"
    provider = "etherscan"
    description = "Ethereum blockchain explorer API — query transactions, balances, tokens, and contract ABIs."
    tags = ["web3"]
    install_hint = ""
    auth_hint = "Set ETHERSCAN_API_KEY environment variable"
    url = "https://etherscan.io/apis"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        key = os.environ.get("ETHERSCAN_API_KEY", "")
        if not key:
            return r
        r.available = True
        r.authenticated = True
        r.auth_method = "env_var"
        r.detail = "key set (env)"
        r.capabilities = ["transactions", "balances", "contracts", "tokens"]
        return r


# ── CoinGecko API ────────────────────────────────────────────────────

class CoinGeckoProbe(HostProbe):
    id = "host_coingecko"
    name = "CoinGecko API"
    provider = "coingecko"
    description = "Crypto market data — prices, trading volume, market cap, and historical charts for 10,000+ coins."
    tags = ["web3"]
    install_hint = ""
    auth_hint = "Set COINGECKO_API_KEY for Pro tier (free tier works without)"
    url = "https://www.coingecko.com/en/api"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        key = os.environ.get("COINGECKO_API_KEY", "")
        # CoinGecko has a free tier without key, but we check for the pro key
        if key:
            r.available = True
            r.authenticated = True
            r.auth_method = "env_var"
            r.capabilities = ["prices", "markets", "history", "exchanges"]
        else:
            r.available = True
            r.authenticated = True
            r.auth_method = "none"
            r.detail = "free tier (rate limited)"
            r.capabilities = ["prices", "markets"]
        return r


# ── IPFS ─────────────────────────────────────────────────────────────

class IpfsProbe(HostProbe):
    id = "host_ipfs"
    name = "IPFS"
    provider = "ipfs"
    description = "Decentralized file storage — content-addressed, peer-to-peer file sharing and hosting."
    tags = ["web3"]
    install_hint = "Download IPFS Desktop or Kubo CLI"
    auth_hint = "Run: ipfs init"
    url = "https://docs.ipfs.tech"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("ipfs"):
            return r
        r.available = True

        code, out, _ = await self.run_cmd("ipfs", "version")
        if code == 0:
            m = re.search(r"(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        code, out, _ = await self.run_cmd("ipfs", "id", "-f", "<id>")
        if code == 0 and out.strip():
            r.authenticated = True
            r.auth_method = "none"
            r.account = out.strip()[:16] + "..."
            r.capabilities = ["add", "pin", "cat", "dag"]

        return r


# ── Slack CLI / Bot ──────────────────────────────────────────────────

class SlackProbe(HostProbe):
    id = "host_slack"
    name = "Slack"
    provider = "slack"
    description = "Team messaging platform — channels, threads, file sharing, and workflow automations."
    tags = ["comms", "office"]
    install_hint = ""
    auth_hint = "Set SLACK_BOT_TOKEN environment variable"
    url = "https://api.slack.com/apps"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        token = os.environ.get("SLACK_BOT_TOKEN") or os.environ.get("SLACK_TOKEN")
        if not token:
            return r
        r.available = True
        r.authenticated = True
        r.auth_method = "env_var"
        r.detail = "bot token set"
        r.capabilities = ["messages", "channels", "files", "reactions"]
        return r


# ── Discord Bot ──────────────────────────────────────────────────────

class DiscordProbe(HostProbe):
    id = "host_discord"
    name = "Discord"
    provider = "discord"
    description = "Community platform — bots, voice channels, rich embeds, and server management automation."
    tags = ["comms"]
    install_hint = ""
    auth_hint = "Set DISCORD_BOT_TOKEN environment variable"
    url = "https://discord.com/developers/applications"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        token = os.environ.get("DISCORD_BOT_TOKEN") or os.environ.get("DISCORD_TOKEN")
        if not token:
            return r
        r.available = True
        r.authenticated = True
        r.auth_method = "env_var"
        r.detail = "bot token set"
        r.capabilities = ["messages", "embeds", "voice", "roles"]
        return r


# ── Notion API ───────────────────────────────────────────────────────

class NotionProbe(HostProbe):
    id = "host_notion"
    name = "Notion"
    provider = "notion"
    description = "All-in-one workspace — docs, wikis, databases, and project tracking with a powerful API."
    tags = ["office"]
    install_hint = ""
    auth_hint = "Set NOTION_API_KEY environment variable"
    url = "https://developers.notion.com"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        key = os.environ.get("NOTION_API_KEY") or os.environ.get("NOTION_TOKEN")
        if not key:
            return r
        r.available = True
        r.authenticated = True
        r.auth_method = "env_var"
        r.detail = "integration token set"
        r.capabilities = ["pages", "databases", "blocks", "search"]
        return r


# ── Figma API ────────────────────────────────────────────────────────

class FigmaProbe(HostProbe):
    id = "host_figma"
    name = "Figma"
    provider = "figma"
    description = "Design platform API — export assets, inspect components, and sync design tokens to code."
    tags = ["design", "dev"]
    install_hint = ""
    auth_hint = "Set FIGMA_TOKEN environment variable"
    url = "https://www.figma.com/developers"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        key = os.environ.get("FIGMA_TOKEN") or os.environ.get("FIGMA_ACCESS_TOKEN")
        if not key:
            return r
        r.available = True
        r.authenticated = True
        r.auth_method = "env_var"
        r.detail = "access token set"
        r.capabilities = ["files", "components", "images", "comments"]
        return r


# ── Supabase ─────────────────────────────────────────────────────────

class SupabaseProbe(HostProbe):
    id = "host_supabase"
    name = "Supabase"
    provider = "supabase"
    description = "Open-source Firebase alternative — Postgres database, auth, storage, and realtime subscriptions."
    tags = ["dev", "data", "cloud"]
    install_hint = "npm install -g supabase"
    auth_hint = "Set SUPABASE_URL and SUPABASE_KEY environment variables"
    url = "https://supabase.com"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        has_cli = self.has_command("supabase")
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

        if not has_cli and not url:
            return r
        r.available = True

        if has_cli:
            code, out, _ = await self.run_cmd("supabase", "--version")
            if code == 0:
                m = re.search(r"(\d+\.\d+\.\d+)", out)
                if m:
                    r.version = m.group(1)

        if url and key:
            r.authenticated = True
            r.auth_method = "env_var"
            r.detail = "project URL and key set"

        r.capabilities = ["database", "auth", "storage", "realtime", "functions"]
        return r


# ── Stripe API ───────────────────────────────────────────────────────

class StripeProbe(HostProbe):
    id = "host_stripe"
    name = "Stripe"
    provider = "stripe"
    description = "Payment processing API — subscriptions, invoices, checkout, and financial reporting."
    tags = ["office"]
    install_hint = "winget install Stripe.StripeCLI"
    auth_hint = "Set STRIPE_SECRET_KEY environment variable"
    url = "https://stripe.com/docs/stripe-cli"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        key = os.environ.get("STRIPE_SECRET_KEY") or os.environ.get("STRIPE_API_KEY")
        has_cli = self.has_command("stripe")

        if not has_cli and not key:
            return r
        r.available = True

        if has_cli:
            code, out, _ = await self.run_cmd("stripe", "version")
            if code == 0:
                m = re.search(r"(\d+\.\d+\.\d+)", out)
                if m:
                    r.version = m.group(1)

        if key:
            r.authenticated = True
            r.auth_method = "env_var"
            mode = "live" if key.startswith("sk_live") else "test"
            r.detail = f"{mode} mode key set"

        r.capabilities = ["payments", "subscriptions", "invoices", "webhooks"]
        return r


# ── Twilio API ───────────────────────────────────────────────────────

class TwilioProbe(HostProbe):
    id = "host_twilio"
    name = "Twilio"
    provider = "twilio"
    description = "Communications API — send SMS, make calls, video chat, and WhatsApp messaging programmatically."
    tags = ["comms", "office"]
    install_hint = ""
    auth_hint = "Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN environment variables"
    url = "https://www.twilio.com/docs"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        sid = os.environ.get("TWILIO_ACCOUNT_SID")
        token = os.environ.get("TWILIO_AUTH_TOKEN")

        if not sid:
            return r
        r.available = True

        if sid and token:
            r.authenticated = True
            r.auth_method = "env_var"
            r.account = sid[:8] + "..."
            r.detail = "account SID and auth token set"

        r.capabilities = ["sms", "voice", "video", "whatsapp"]
        return r


# ── SendGrid API ─────────────────────────────────────────────────────

class SendGridProbe(HostProbe):
    id = "host_sendgrid"
    name = "SendGrid"
    provider = "twilio"
    description = "Transactional email API — send receipts, alerts, marketing campaigns, and track delivery."
    tags = ["comms", "office"]
    install_hint = ""
    auth_hint = "Set SENDGRID_API_KEY environment variable"
    url = "https://docs.sendgrid.com"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        key = os.environ.get("SENDGRID_API_KEY")
        if not key:
            return r
        r.available = True
        r.authenticated = True
        r.auth_method = "env_var"
        r.detail = "API key set"
        r.capabilities = ["email", "templates", "analytics"]
        return r


# ── Helm ─────────────────────────────────────────────────────────────

class HelmProbe(HostProbe):
    id = "host_helm"
    name = "Helm"
    provider = "helm"
    description = "Kubernetes package manager — install, upgrade, and manage complex application deployments."
    tags = ["devops"]
    install_hint = "winget install Helm.Helm"
    auth_hint = ""
    url = "https://helm.sh"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("helm"):
            return r
        r.available = True
        r.authenticated = True
        r.auth_method = "none"

        code, out, _ = await self.run_cmd("helm", "version", "--short")
        if code == 0:
            m = re.search(r"v?(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        r.capabilities = ["charts", "releases", "repos", "rollback"]
        return r


# ── Ansible ──────────────────────────────────────────────────────────

class AnsibleProbe(HostProbe):
    id = "host_ansible"
    name = "Ansible"
    provider = "redhat"
    description = "Agentless automation — configure servers, deploy apps, and orchestrate IT workflows with YAML."
    tags = ["devops"]
    install_hint = "pip install ansible"
    auth_hint = ""
    url = "https://docs.ansible.com"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("ansible"):
            return r
        r.available = True
        r.authenticated = True
        r.auth_method = "none"

        code, out, _ = await self.run_cmd("ansible", "--version")
        if code == 0:
            m = re.search(r"(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        r.capabilities = ["playbooks", "roles", "inventory", "vault"]
        return r


# ── FFmpeg ───────────────────────────────────────────────────────────

class FfmpegProbe(HostProbe):
    id = "host_ffmpeg"
    name = "FFmpeg"
    provider = "ffmpeg"
    description = "Media processing toolkit — convert, stream, record, and manipulate audio and video files."
    tags = ["dev", "design"]
    install_hint = "winget install Gyan.FFmpeg"
    auth_hint = ""
    url = "https://ffmpeg.org"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("ffmpeg"):
            return r
        r.available = True
        r.authenticated = True
        r.auth_method = "none"

        code, out, err = await self.run_cmd("ffmpeg", "-version")
        combined = out + err
        m = re.search(r"ffmpeg version (\S+)", combined)
        if m:
            r.version = m.group(1)

        r.capabilities = ["transcode", "stream", "record", "filters"]
        return r


# ── ImageMagick ──────────────────────────────────────────────────────

class ImageMagickProbe(HostProbe):
    id = "host_imagemagick"
    name = "ImageMagick"
    provider = "imagemagick"
    description = "Image processing from the command line — resize, crop, convert, and compose images in bulk."
    tags = ["design", "dev"]
    install_hint = "winget install ImageMagick.ImageMagick"
    auth_hint = ""
    url = "https://imagemagick.org"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("magick") and not self.has_command("convert"):
            return r
        r.available = True
        r.authenticated = True
        r.auth_method = "none"

        cmd = "magick" if self.has_command("magick") else "convert"
        code, out, _ = await self.run_cmd(cmd, "--version")
        if code == 0:
            m = re.search(r"(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        r.capabilities = ["convert", "resize", "composite", "animate"]
        return r


# ── Vault (HashiCorp) ────────────────────────────────────────────────

class VaultProbe(HostProbe):
    id = "host_vault"
    name = "Vault"
    provider = "hashicorp"
    description = "Secrets management — store API keys, passwords, and certificates with access control policies."
    tags = ["security", "devops"]
    install_hint = "winget install Hashicorp.Vault"
    auth_hint = "Set VAULT_ADDR and VAULT_TOKEN environment variables"
    url = "https://www.vaultproject.io"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("vault"):
            return r
        r.available = True

        code, out, _ = await self.run_cmd("vault", "version")
        if code == 0:
            m = re.search(r"v(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        addr = os.environ.get("VAULT_ADDR")
        token = os.environ.get("VAULT_TOKEN")
        if addr and token:
            r.authenticated = True
            r.auth_method = "env_var"
            r.detail = f"server: {addr}"
            r.capabilities = ["secrets", "kv", "transit", "pki"]

        return r


# ── 1Password CLI ────────────────────────────────────────────────────

class OnePasswordProbe(HostProbe):
    id = "host_1password"
    name = "1Password"
    provider = "1password"
    description = "Password manager CLI — access vaults, inject secrets into scripts, and manage team credentials."
    tags = ["security"]
    install_hint = "winget install AgileBits.1Password.CLI"
    auth_hint = "Run: op signin"
    url = "https://developer.1password.com/docs/cli"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("op"):
            return r
        r.available = True

        code, out, _ = await self.run_cmd("op", "--version")
        if code == 0:
            m = re.search(r"(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        code, out, _ = await self.run_cmd("op", "whoami")
        if code == 0 and out.strip():
            r.authenticated = True
            r.auth_method = "cli"
            r.account = out.strip().splitlines()[0] if out.strip() else ""
            r.capabilities = ["vaults", "items", "secrets", "ssh_agent"]

        return r


# ── Fly.io ───────────────────────────────────────────────────────────

class FlyProbe(HostProbe):
    id = "host_fly"
    name = "Fly.io"
    provider = "fly"
    description = "Edge application platform — deploy full-stack apps close to users with built-in Postgres and Redis."
    tags = ["cloud", "devops"]
    install_hint = "winget install Flyctl"
    auth_hint = "Run: fly auth login"
    url = "https://fly.io/docs"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("flyctl") and not self.has_command("fly"):
            return r
        r.available = True

        cmd = "flyctl" if self.has_command("flyctl") else "fly"
        code, out, _ = await self.run_cmd(cmd, "version")
        if code == 0:
            m = re.search(r"(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        code, out, _ = await self.run_cmd(cmd, "auth", "whoami")
        if code == 0 and out.strip() and "not logged in" not in out.lower():
            r.authenticated = True
            r.auth_method = "cli"
            r.account = out.strip()
            r.capabilities = ["apps", "machines", "volumes", "postgres", "redis"]

        return r


# ── Cloudflare (Wrangler) ────────────────────────────────────────────

class CloudflareProbe(HostProbe):
    id = "host_cloudflare"
    name = "Cloudflare"
    provider = "cloudflare"
    description = "Edge platform — Workers serverless functions, R2 storage, D1 database, and DNS management."
    tags = ["cloud", "dev"]
    install_hint = "npm install -g wrangler"
    auth_hint = "Run: wrangler login, or set CLOUDFLARE_API_TOKEN"
    url = "https://developers.cloudflare.com/workers/wrangler/"

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        if not self.has_command("wrangler"):
            return r
        r.available = True

        code, out, _ = await self.run_cmd("wrangler", "--version")
        if code == 0:
            m = re.search(r"(\d+\.\d+\.\d+)", out)
            if m:
                r.version = m.group(1)

        code, out, _ = await self.run_cmd("wrangler", "whoami")
        if code == 0 and out.strip() and "not authenticated" not in out.lower():
            r.authenticated = True
            r.auth_method = "cli"
            r.capabilities = ["workers", "r2", "d1", "kv", "dns"]

        if not r.authenticated and os.environ.get("CLOUDFLARE_API_TOKEN"):
            r.authenticated = True
            r.auth_method = "env_var"
            r.capabilities = ["workers", "r2", "d1", "kv", "dns"]

        return r


# ── Environment credentials (names only) ─────────────────────────────

# Curated well-known credential env vars. Presence of any of these is a strong
# signal the owner has a real credential in the environment they may want to
# import into the vault. NAMES ONLY are ever surfaced.
_WELL_KNOWN_CRED_ENV = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "BRAVE_API_KEY",
    "STRIPE_SECRET_KEY",
    "TWILIO_AUTH_TOKEN",
    "SENDGRID_API_KEY",
    "ETHERSCAN_API_KEY",
    "PYPI_API_TOKEN",
    "NPM_TOKEN",
    "CLOUDFLARE_API_TOKEN",
    "TELEGRAM_BOT_TOKEN",
    "DISCORD_BOT_TOKEN",
    "SLACK_BOT_TOKEN",
    "NOTION_API_KEY",
)

# Generic credential-shaped name suffixes. A remaining env var whose NAME
# matches and whose VALUE is a non-empty string of length >= 8 is treated as a
# likely credential.
_CRED_NAME_RE = re.compile(r"(?i)(_API_KEY|_TOKEN|_SECRET|_PASSWORD|_PASSWD|_CREDENTIALS?)$")

# Prefixes / names that look credential-shaped but are NOT owner secrets we want
# to surface: autonet's own config namespace, keystore internals, and anything
# resembling a PATH list.
_CRED_EXCLUDE_PREFIXES = ("ATN_", "KEYSTORE_")
_CRED_EXCLUDE_NAMES = frozenset({"PATH", "PYTHONPATH", "CLASSPATH", "LD_LIBRARY_PATH"})


class EnvCredentialProbe(HostProbe):
    id = "host_env_credentials"
    name = "Environment credentials"
    provider = "environment"
    description = ("Credential-shaped environment variables detected on the "
                   "host — names only, never values. Import into the vault to "
                   "hand them to agents under owner control.")
    tags = ["security"]
    install_hint = ""
    auth_hint = ""
    url = ""

    @staticmethod
    def _looks_like_path(name: str, value: str) -> bool:
        upper = name.upper()
        if upper in _CRED_EXCLUDE_NAMES or upper.endswith("PATH"):
            return True
        # Heuristic: a value carrying an OS path separator list is not a secret.
        return (os.pathsep in value and (os.sep in value or "/" in value))

    def _scan(self) -> list[str]:
        env = os.environ
        found: list[str] = []
        seen: set[str] = set()

        # Tier 1 — curated well-known names.
        for name in _WELL_KNOWN_CRED_ENV:
            val = env.get(name)
            if isinstance(val, str) and val:
                found.append(name)
                seen.add(name)

        # Tier 2 — generic pattern sweep over the remaining environment.
        for name, val in env.items():
            if name in seen:
                continue
            if any(name.startswith(p) for p in _CRED_EXCLUDE_PREFIXES):
                continue
            if not _CRED_NAME_RE.search(name):
                continue
            if not isinstance(val, str) or len(val) < 8:
                continue
            if self._looks_like_path(name, val):
                continue
            found.append(name)
            seen.add(name)

        return found

    async def probe(self) -> ProbeResult:
        r = ProbeResult(id=self.id, name=self.name, provider=self.provider)
        names = self._scan()
        r.env_keys = names
        r.available = bool(names)
        if names:
            r.authenticated = True
            r.auth_method = "env_var"
            n = len(names)
            r.detail = (f"{n} credential-shaped variable"
                        f"{'s' if n != 1 else ''} in the environment")
            r.capabilities = ["import"]
        return r


# ── All probes ────────────────────────────────────────────────────────

ALL_PROBES: list[type[HostProbe]] = [
    # Dev fundamentals
    GitProbe,
    GitHubProbe,
    NodeProbe,
    PythonProbe,
    RustProbe,
    GoProbe,
    JavaProbe,
    DockerProbe,

    # Cloud platforms
    AwsProbe,
    GcpProbe,
    AzureProbe,

    # DevOps & hosting
    KubernetesProbe,
    HelmProbe,
    TerraformProbe,
    AnsibleProbe,
    NetlifyProbe,
    VercelProbe,
    FirebaseProbe,
    FlyProbe,
    CloudflareProbe,
    SupabaseProbe,

    # AI / ML
    AnthropicKeyProbe,
    OpenAiKeyProbe,
    OllamaProbe,
    HuggingFaceProbe,

    # Data
    PostgresProbe,
    RedisProbe,
    MongoProbe,
    SqliteProbe,

    # Web3
    FoundryProbe,
    EtherscanProbe,
    CoinGeckoProbe,
    IpfsProbe,

    # Security
    SshProbe,
    GpgProbe,
    VaultProbe,
    OnePasswordProbe,
    EnvCredentialProbe,

    # Communications
    SlackProbe,
    DiscordProbe,
    TwilioProbe,
    SendGridProbe,

    # Office / SaaS
    NotionProbe,
    StripeProbe,

    # Design & media
    FigmaProbe,
    FfmpegProbe,
    ImageMagickProbe,
]

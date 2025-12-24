# Security Policy

## Supported Versions

Currently supported versions of loom-agent:

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3.0 | :x:                |

## Reporting a Vulnerability

We take the security of loom-agent seriously. If you discover a security vulnerability, please follow these steps:

### How to Report

1. **Do NOT** open a public issue
2. Email the maintainers at: wanghaishan0210@gmail.com
3. Include the following information:
   - Description of the vulnerability
   - Steps to reproduce the issue
   - Potential impact
   - Suggested fix (if available)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours
- **Updates**: We will provide regular updates on our progress (at least every 7 days)
- **Timeline**: We aim to release a fix within 30 days for critical vulnerabilities
- **Credit**: We will credit you in the security advisory (unless you prefer to remain anonymous)

### Security Best Practices for Users

When using loom-agent:

1. **API Keys**: Never commit API keys or credentials to version control
2. **Dependencies**: Regularly update dependencies to get security patches
3. **Input Validation**: Always validate and sanitize user inputs before processing
4. **Production**: Use the latest stable version in production environments
5. **Environment Variables**: Store sensitive configuration in environment variables

### Known Security Considerations

- LLM Provider security: When using LLM providers (OpenAI, Anthropic, etc.), ensure you follow their security guidelines
- Redis/NATS: If using Redis or NATS transport layers, ensure they are properly secured with authentication
- Memory Storage: Be cautious about storing sensitive information in memory systems

## Security Updates

Security updates will be announced in:
- GitHub Security Advisories
- CHANGELOG.md
- Release notes

## Contact

For security-related questions: wanghaishan0210@gmail.com
For general questions: Open a GitHub issue

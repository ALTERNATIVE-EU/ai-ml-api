# Token Management with CKAN

## Integration with CKAN

The integration is achieved through a combination of custom development and configuration, ensuring a seamless experience for both administrators and end-users.

### Custom CKAN Extension

A custom CKAN extension has been developed to bridge CKAN with Keycloak, enabling advanced token management features. This extension allows for:

- Seamless synchronization between CKAN user accounts and Keycloak authentication.
- Enhanced security measures through Keycloak's robust authentication mechanisms.

### Database Schema

The CKAN database schema has been extended to support the storage of token metadata, including:

- Token identifiers and associated user accounts.
- Token scopes, expiration dates, and creation timestamps.

## User Interface Enhancements

To improve the user experience, we have introduced new UI components focused on token management within the CKAN platform.

<!-- ![Token Management UI](path/to/token_management_ui.png) -->

### New UI Components

- **Token Management Dashboard**: Integrated into the user profile, this dashboard provides a comprehensive overview of a user's tokens, including creation dates, scopes, and expiration dates.
- **Token Creation Wizard**: A user-friendly interface that guides users through the process of creating new tokens, with options to customize scopes and set expiration dates.
- **Token Revocation and Renewal Interface**: Easy-to-use interface that allow users to revoke or renew their tokens.

### Accessibility

In line with our commitment to inclusivity, the new UI components are designed to be accessible to all users, ensuring a seamless experience regardless of individual needs or preferences.
# Deployment Process
## Production Deployment: 

For the production deployment, we use Kubernetes to manage the deployment of the AI/ML API server and its associated components. The deployment process involves the following steps:

### AI/ML API Server Deployment:
1. Build Docker Image
2. Push Image to Registry
3. Update Deployment Files
4. Apply Deployment

In the project README, we provide detailed instructions for each step of the deployment process, including the commands to execute.

### Custom Envoy Filter Deployment:
1. Apply the SQL Schema
2. Build the Filter
3. Create PVC
4. Add volume and volume mount to the sidecar container
5. Copy the filter binary to the volume
6. Update the Envoy Filter manifest configuration
7. Apply the Envoy Filter manifest
## Setup Terraform on AWS

1. Create IAM Role:

Terraform needs full access to all required AWS services. Therefore, create a IAM role that can be inferred by Terraform:
- TerraformServiceRole:
  - Permission Policies: AdministratorAccess
  - Trust Relationship: see json file "terraform_iam_role_trust_relationship.json"

2. Create User:

The created IAM role needs to be attached to a user who is member of a user group.
- User Group: CICD
- User: Gitlab
  - Permission Policies: gitlab_permissions (see json file "gitlab_permissions.json")

3. Create TF-State S3 Bucket:

In order to make the Terraform deployment available for an organisation the Terraform state file must be persisted in a accessable S3 bucket.
- S3 Bucket: xxx-ml-ops-tfstate-production


## Prerequisites of Kubeflow Deployment

1. Provision Domain:

In order to make Kubeflow available via the internet a public domain is needed. AWS offers a service called Route53 which can be used for provision domain.
- Route53 > Hosted zones > Create hosted zone

2. Setup Gitlab-CI:

The following configurations must be made so that Teraform can be executed via CI/CD:
- Create Gitlab Variables: Settings > CI/CD > Variables
  - AWS_ACCESS_KEY_ID (mask variable, expand variable reference)
  - AWS_ACCOUNT_ID (mask variable, expand variable reference)
  - AWS_SECRET_ACCESS_KEY (mask variable, expand variable reference)
  - AWS_REGION (mask variable, expand variable reference)
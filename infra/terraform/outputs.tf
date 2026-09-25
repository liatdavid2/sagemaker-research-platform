output "artifact_bucket" {
  value = aws_s3_bucket.artifacts.bucket
}

output "sagemaker_role_arn" {
  value = aws_iam_role.sagemaker_execution.arn
}

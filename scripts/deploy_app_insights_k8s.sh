# Specify the Istio version that will be leveraged throughout these instructions
ISTIO_VERSION=1.4.0

curl -sL "https://github.com/istio/istio/releases/download/$ISTIO_VERSION/istio-$ISTIO_VERSION-linux.tar.gz" | tar xz

cd istio-$ISTIO_VERSION
sudo cp ./bin/istioctl /usr/local/bin/istioctl
sudo chmod +x /usr/local/bin/istioctl

# Generate the bash completion file and source it in your current shell
mkdir -p ~/completions && istioctl collateral --bash -o ~/completions
source ~/completions/istioctl.bash

# Source the bash completion file in your .bashrc so that the command-line completions
# are permanently available in your shell
echo "source ~/completions/istioctl.bash" >> ~/.bashrc

kubectl create namespace istio-system --save-config

#apiVersion: install.istio.io/v1alpha2
#kind: IstioControlPlane
#spec:
#  # Use the default profile as the base
#  # More details at: https://istio.io/docs/setup/additional-setup/config-profiles/
#  profile: default
#  values:
#    global:
#      # Ensure that the Istio pods are only scheduled to run on Linux nodes
#      defaultNodeSelector:
#        beta.kubernetes.io/os: linux
#      # Enable mutual TLS for the control plane
#      controlPlaneSecurityEnabled: true
#      mtls:
#        # Require all service to service communication to have mtls
#        enabled: false
#    grafana:
#      # Enable Grafana deployment for analytics and monitoring dashboards
#      enabled: true
#      security:
#        # Enable authentication for Grafana
#        enabled: true
#    kiali:
#      # Enable the Kiali deployment for a service mesh observability dashboard
#      enabled: true
#    tracing:
#      # Enable the Jaeger deployment for tracing
#      enabled: true

istioctl manifest apply -f istio.aks.yaml --logtostderr --set installPackagePath=./install/kubernetes/operator/charts

kubectl get svc --namespace istio-system --output wide

kubectl label namespace <my-app-namespace> istio-injection=enabled
kubectl apply -f .
kubectl get pods -n istio-system -l "app=application-insights-istio-mixer-adapter"
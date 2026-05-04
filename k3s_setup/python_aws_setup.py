# aws_cluster_setup.py
import boto3
import time
import os
import paramiko

# --- CẤU HÌNH ---
REGION = "us-east-1" # Đổi nếu Learner Lab của bạn ở region khác
KEY_NAME = "k8s-doan-key-auto"
SG_NAME = "k8s-cluster-sg-auto"

print("🚀 KHỞI ĐỘNG HỆ THỐNG XÂY DỰNG HẠ TẦNG K8S TỰ ĐỘNG...")
ec2 = boto3.client('ec2', region_name=REGION)
ec2_resource = boto3.resource('ec2', region_name=REGION)

# 1. TẠO HOẶC LẤY SECURITY GROUP
print("\n[1/5] Đang thiết lập Security Group (Mở cổng mạng)...")
vpcs = ec2.describe_vpcs()
vpc_id = vpcs['Vpcs'][0]['VpcId']

try:
    sg = ec2.create_security_group(GroupName=SG_NAME, Description='K8s Doan Auto SG', VpcId=vpc_id)
    sg_id = sg['GroupId']
    # Mở full traffic nội bộ và cổng SSH, 6443 từ bên ngoài
    ec2.authorize_security_group_ingress(
        GroupId=sg_id,
        IpPermissions=[
            {'IpProtocol': '-1', 'FromPort': -1, 'ToPort': -1, 'IpRanges': [{'CidrIp': '0.0.0.0/0'}]}
        ]
    )
    print(f"✅ Đã tạo Security Group mới: {sg_id}")
except Exception as e:
    sgs = ec2.describe_security_groups(GroupNames=[SG_NAME])
    sg_id = sgs['SecurityGroups'][0]['GroupId']
    print(f"⚡ Security Group đã tồn tại: {sg_id}")

# 2. TẠO HOẶC LẤY KEY PAIR
print("\n[2/5] Đang thiết lập SSH Key Pair...")
key_path = f"./{KEY_NAME}.pem"
try:
    key_pair = ec2.create_key_pair(KeyName=KEY_NAME)
    with open(key_path, "w") as file:
        file.write(key_pair['KeyMaterial'])
    os.chmod(key_path, 0o400) # Cấp quyền read-only bảo mật
    print(f"✅ Đã tạo khóa mới và lưu tại: {key_path}")
except Exception as e:
    print(f"⚡ Khóa {KEY_NAME} đã tồn tại trên AWS. Đảm bảo bạn đang giữ file .pem nhé!")

# Lấy ID của Ubuntu 24.04/22.04 mới nhất
ami_response = ec2.describe_images(
    Filters=[{'Name': 'name', 'Values': ['ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*']}],
    Owners=['099720109477'] # Owner ID của Canonical
)
ami_id = sorted(ami_response['Images'], key=lambda x: x['CreationDate'], reverse=True)[0]['ImageId']

# 3. TẠO MASTER NODE
print("\n[3/5] Đang cấp phát Master Node (t3.small)...")
master_instance = ec2_resource.create_instances(
    ImageId=ami_id, MinCount=1, MaxCount=1,
    InstanceType='t3.small', KeyName=KEY_NAME,
    SecurityGroupIds=[sg_id],
    TagSpecifications=[{'ResourceType': 'instance', 'Tags': [{'Key': 'Name', 'Value': 'Master-Node-Auto'}]}]
)[0]

print("⏳ Chờ Master Node khởi động (khoảng 30 giây)...")
master_instance.wait_until_running()
master_instance.reload()
master_public_ip = master_instance.public_ip_address
master_private_ip = master_instance.private_ip_address
print(f"✅ Master Node đang chạy! Public IP: {master_public_ip} | Private IP: {master_private_ip}")

# 4. CÀI ĐẶT K3S VÀ LẤY TOKEN (Qua SSH)
print("\n[4/5] Đang kết nối SSH vào Master để cài đặt K3s (Có thể mất 1-2 phút)...")
time.sleep(30) # Chờ SSH service trên máy ảo sẵn sàng

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(hostname=master_public_ip, username='ubuntu', key_filename=key_path)

print("⚙️ Đang cài đặt K3s...")
stdin, stdout, stderr = ssh.exec_command('curl -sfL https://get.k3s.io | sh -')
stdout.channel.recv_exit_status() # Đợi lệnh chạy xong

print("🔑 Đang lấy Node Token và Kubeconfig...")
stdin, stdout, stderr = ssh.exec_command('sudo cat /var/lib/rancher/k3s/server/node-token')
node_token = stdout.read().decode('utf-8').strip()

stdin, stdout, stderr = ssh.exec_command('sudo cat /etc/rancher/k3s/k3s.yaml')
kubeconfig_data = stdout.read().decode('utf-8')
ssh.close()

# Lưu config về máy và đổi IP
kubeconfig_data = kubeconfig_data.replace('127.0.0.1', master_public_ip)
with open("./config_auto.yaml", "w") as f:
    f.write(kubeconfig_data)
print("✅ Đã lưu file cấu hình K8s tại ./config_auto.yaml")

# 5. TẠO 4 WORKER NODES VỚI USER DATA
print("\n[5/5] Đang cấp phát 4 Worker Nodes và tự động join vào cụm (t2.micro)...")
worker_user_data = f"""#!/bin/bash
curl -sfL https://get.k3s.io | K3S_URL=https://{master_private_ip}:6443 K3S_TOKEN={node_token} sh -
"""

workers = ec2_resource.create_instances(
    ImageId=ami_id, MinCount=4, MaxCount=4,
    InstanceType='t2.micro', KeyName=KEY_NAME,
    SecurityGroupIds=[sg_id],
    UserData=worker_user_data,
    TagSpecifications=[{'ResourceType': 'instance', 'Tags': [{'Key': 'Name', 'Value': 'Worker-Node-Auto'}]}]
)

print("⏳ Cụm đang được định hình. Các Worker sẽ tự động Join vào Master trong khoảng 1 phút nữa.")
print("="*60)
print("🎉 HOÀN TẤT SETUP! 🎉")
print("Bây giờ bạn có thể kiểm tra cụm bằng lệnh:")
print("kubectl --kubeconfig=./config_auto.yaml get nodes")
print("="*60)
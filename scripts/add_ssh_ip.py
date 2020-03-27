from azure.common.client_factory import get_client_from_cli_profile
from azure.mgmt.network import NetworkManagementClient
import argparse
import logging
import requests
import time


logger = logging.getLogger(__name__)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Add IP to SSH whitelist in CleanupTool Security Rule")
    parser.add_argument("resource_group", help="Resource Group Name")
    parser.add_argument("--ip", "-i", help="Manually specified IP to add")
    parser.add_argument("--repeat", "-r", action="store_true", help="Repeat action every minute")
    parser.add_argument("--quiet", "-q", action="store_true", help="Run in quiet mode without logging info")
    args = parser.parse_args()

    level=logging.ERROR if args.quiet else logging.INFO
    logging.basicConfig(level=level)

    # get current extenal ip
    ip = args.ip or requests.get('https://api.ipify.org').text
    logger.info("Using IP Address: %s", ip)

    # get network mgmt client using cli credentials
    logger.info("Starting Network Management Client")
    client = get_client_from_cli_profile(NetworkManagementClient)

    try:
        # get network security group name for resource group
        vnet = client.virtual_networks.list(args.resource_group).next()
        nsg_name = vnet.subnets[0].network_security_group.id.split('/')[-1]
        logger.info("Found Network Security Group: %s", nsg_name)
    except:
        logger.error("Could not find Network Security Group, make sure resources are correct and initialized", exc_info=True)
        exit

    repeat = True
    while repeat:
        # get security rule for allowing inbound ssh connections
        security_rule = client.security_rules.get('cleanupservice', nsg_name, 'CleanupTool-Allow-100')

        # get list of allowed ips
        allowed_ips = security_rule.source_address_prefixes

        if ip not in allowed_ips:
            # add current ip to list
            security_rule.source_address_prefixes = [ip] + allowed_ips
        
            try:
                # update the security rule
                logger.info("Updating SSH Security Rule")
                client.security_rules.create_or_update('cleanupservice', nsg_name, 'CleanupTool-Allow-100', security_rule)
                logger.info("Security Rule Updated: please wait a few seconds before attempting to connect")
            except:
                # TODO: catch update in progress error
                pass

        repeat = args.repeat
        if repeat:
            time.sleep(60)


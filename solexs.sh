#! /bin/bash
#Sample bash script to automate data download via PRADAN. 
#Windows users may install wget.exe and write a batch script in the same lines.
#Prequisites: Login to Pradan in your browser, select data of your interest and download script for the session
#Caution: There are session download limits, request rate limit and session timeouts in place, etc.
#	Violations may lead to blocking. Use script to ease the manual data download efforts but do not load the server.

cookies="FGTServer=03DE191863F4388C06A7AAAF7E0136FBD15060DF21FA637D82A675307CD5BF28BF8658CAFD950178C9994D;FGTServer=03DE191863F4388C06A7AAAF7E0136FBD15060DF21FA637D82A675307CD5BF28BF8658CAFD950178C9994D;JSESSIONID=aa65b5546a91cf48c4304f75c09b;FGTServer=03DE191863F4388C06A7AAAF7E0136FBD15060DF21FA637D82A675307CD5BF28BF8658CAFD950178C9994D;JSESSIONID=aa65482540695f403abc4503b47d;OAuth_Token_Request_State=15008e9c-2c12-41df-84fe-dc5426151f0c;"
urlPrefix="https://pradan1.issdc.gov.in"
#proxyOptions are required if your organization uses proxy to connect to Internet.
#proxyOptions="-e use_proxy=yes -e https_proxy=127.0.0.1:8080"
proxyOptions=""

#keepalive for 1 day max
counter=144;while [ $counter -gt 0 ]; do sleep 10m; wget $proxyOptions -N --content-disposition --tries=1 --no-cookies --header "Cookie: $cookies" $urlPrefix"/al1/protected/payload.xhtml"; counter=$(($counter-1)); done &
bdpid=$!

dataFilePaths=("/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260622_v1.0.zip?solexs" "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260621_v1.0.zip?solexs" "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260620_v1.0.zip?solexs" "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260619_v1.0.zip?solexs" "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260618_v1.0.zip?solexs" "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260617_v1.0.zip?solexs" "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260616_v1.0.zip?solexs" "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260615_v1.0.zip?solexs" "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260614_v1.0.zip?solexs" "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260613_v1.0.zip?solexs" )

i=0;
for file in ${dataFilePaths[@]}
do 
	echo $file; 
	i=$(($i+1));
	wget $proxyOptions -x --max-redirect=0 --content-disposition --tries=1 --no-cookies --header "Cookie: $cookies" $urlPrefix$file;
	if [ $? -ne 0 ]; then
		echo "Error: Limits reached or session expired, terminating without downloading file $i: $file. You may login again later to download script for the new session and resume downloads." 
		kill -9 $bdpid
		exit -1;
	fi
done
echo "Your downloads($i) are complete."

kill -9 $bdpid


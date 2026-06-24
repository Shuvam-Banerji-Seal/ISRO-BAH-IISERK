#! /bin/bash
#Sample bash script to automate data download via PRADAN. 
#Windows users may install wget.exe and write a batch script in the same lines.
#Prequisites: Login to Pradan in your browser, select data of your interest and download script for the session
#Caution: There are session download limits, request rate limit and session timeouts in place, etc.
#	Violations may lead to blocking. Use script to ease the manual data download efforts but do not load the server.

cookies=""
urlPrefix="https://pradan1.issdc.gov.in"
#proxyOptions are required if your organization uses proxy to connect to Internet.
#proxyOptions="-e use_proxy=yes -e https_proxy=127.0.0.1:8080"
proxyOptions=""

#keepalive for 1 day max
counter=144;while [ $counter -gt 0 ]; do sleep 10m; wget $proxyOptions -N --content-disposition --tries=1 --no-cookies --header "Cookie: $cookies" $urlPrefix"/al1/protected/payload.xhtml"; counter=$(($counter-1)); done &
bdpid=$!

dataFilePaths=("/al1/protected/downloadData/hel1os/level1/2026/06/21/N00_0000/HLS_20260621_114950_43804sec_lev1_V111.zip?hel1os" "/al1/protected/downloadData/hel1os/level1/2026/06/21/N00_0000/HLS_20260621_000005_43181sec_lev1_V111.zip?hel1os" "/al1/protected/downloadData/hel1os/level1/2026/06/20/N00_0000/HLS_20260620_121027_42563sec_lev1_V111.zip?hel1os" "/al1/protected/downloadData/hel1os/level1/2026/06/20/N00_0000/HLS_20260620_000008_43177sec_lev1_V111.zip?hel1os" "/al1/protected/downloadData/hel1os/level1/2026/06/16/N00_0000/HLS_20260616_000006_43176sec_lev1_V111.zip?hel1os" "/al1/protected/downloadData/hel1os/level1/2026/06/19/N00_0000/HLS_20260619_115949_43206sec_lev1_V111.zip?hel1os" "/al1/protected/downloadData/hel1os/level1/2026/06/19/N00_0000/HLS_20260619_000005_43182sec_lev1_V111.zip?hel1os" "/al1/protected/downloadData/hel1os/level1/2026/06/18/N00_0000/HLS_20260618_120006_43182sec_lev1_V111.zip?hel1os" "/al1/protected/downloadData/hel1os/level1/2026/06/18/N00_0000/HLS_20260618_000006_43185sec_lev1_V111.zip?hel1os" "/al1/protected/downloadData/hel1os/level1/2026/06/17/N00_0000/HLS_20260617_121028_42562sec_lev1_V111.zip?hel1os" )

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


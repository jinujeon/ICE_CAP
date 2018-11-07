import sys
from sdk.api.message import Message
from sdk.exceptions import CoolsmsException

##  @brief This sample code demonstrate how to send sms through CoolSMS Rest API PHP
if __name__ == "__main__":

    # set api key, api secret
    api_key = "NCSUZ18JACUCCBCU"
    api_secret = "DCI22GGXXXLML7JHDFHJBBSLPIW9N8SK"

    ## 4 params(to, from, type, text) are mandatory. must be filled
    params = dict()
    params['type'] = 'sms'  # Message type ( sms, lms, mms, ata )
    params['to'] = '01043655330'  # Recipients Number '01000000000,01000000001'
    params['from'] = '01043655330'  # Sender number
    params['text'] = 'hello'  # Message

    cool = Message(api_key, api_secret)
    try:
        response = cool.send(params)
        print("Success Count : %s" % response['success_count'])
        print("Error Count : %s" % response['error_count'])
        print("Group ID : %s" % response['group_id'])

        if "error_list" in response:
            print("Error List : %s" % response['error_list'])

    except CoolsmsException as e:
        print("Error Code : %s" % e.code)
        print("Error Message : %s" % e.msg)

    sys.exit()